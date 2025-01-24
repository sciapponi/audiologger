import threading
import time
import whisper 
import torch 
from model.decoder import QuantDecoder

class FIFOBuffer:
    def __init__(self, length):
        self.buf = []
        self.length = length
        self.condition = threading.Condition()  # Condition variable for synchronization

    def enqueue(self, data):
        with self.condition:
            self.buf.append(data)
            self.condition.notify_all()  # Notify any waiting thread(s) that data has been added

    def dequeue(self):
        with self.condition:
            start_time = time.time()  # Record the start time
            while len(self.buf) < self.length:
                # Calculate the remaining time to wait
                elapsed = time.time() - start_time
                remaining = 30 - elapsed
                 # If 10 seconds have passed and the buffer is not empty, break the loop
                if remaining <= 0 and len(self.buf)>0: 
                    break
                self.condition.wait(timeout=remaining)
            
            # Take as much data as is available (up to the required length)
            data = self.buf[:self.length]
            del self.buf[:len(data)]

            return data
        

class WhisperDecodingThreadWrapper(threading.Thread):

    def __init__(self, buffer, decoder, wmodel, language="en"):
        super().__init__()
        # LOAD WHISPER
        self.buf = buffer
        self.wmodel = wmodel
        self.decoder = decoder
        self.running = True
        self.language = language

    def run(self):
        print("run started!")
        while self.running:
            dq = torch.cat(tuple(self.buf.dequeue()), dim=1)
            # print(dq.shape)
            audio_out = self.decoder(dq)
            # print("audio out:", audio_out.shape)
            audio = whisper.pad_or_trim(audio_out)
            mel = whisper.log_mel_spectrogram(audio.squeeze().cuda(), n_mels=128)
            options = whisper.DecodingOptions(language=self.language, task="transcribe")
            result = whisper.decode(self.wmodel, mel, options)
            print("TRANSCRIPTION: \t",result.text)

    def stop(self):
        self.running = False