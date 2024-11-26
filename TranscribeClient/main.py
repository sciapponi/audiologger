import paho.mqtt.client as mqtt
import hydra 
import json 
import sys
from model.decoder import QuantDecoder
import whisper 
import torch 
from utils import FIFOBuffer, WhisperDecodingThreadWrapper

@hydra.main(version_base=None, config_path='./config', config_name='base')
def main(args):

    # LOAD DECODER MODEL
    decoder = QuantDecoder(args.NAC.MODEL_PATH)
    
    # LOAD WHISPER
    wmodel = whisper.load_model("turbo").cuda()
    print("loaded!")
    # Mosquitto MQTT Configuration
    MQTT_BROKER = args.MQTT.BROKER_ADDRESS  # Replace with your MQTT broker address
    MQTT_PORT = 1883
    MQTT_TOPIC = "AUDIOLOGGER"  # Example topic to subscribe to


    # Callback when the client connects to the broker
    def on_connect(client, userdata, flags, rc):
        # print(f"Connected to MQTT Broker with code {rc}")
        client.subscribe(MQTT_TOPIC)

    # Buffer to hold the indices
    buffer = FIFOBuffer(20)
    # Callback when a message is received from the broker
    def on_message(client, userdata, msg):
     # Ensure we can modify the global buffer
        
        try:
            # Decode and parse the message
            payload = msg.payload.decode()
            
            # Convert the payload to a data point
            data = json.loads(payload)  # Safely parse JSON
            indices = data["data"]
            tensor_indices = torch.Tensor(indices).int()
            
            # Append the tensor to the buffer
            buffer.enqueue(tensor_indices)
                
        except Exception as e:
            print(f"Error processing message: {e}")
        except KeyboardInterrupt:
            print("Stream Ended")
            sys.exit(0)

    wthread = WhisperDecodingThreadWrapper(buffer, decoder, wmodel)
    wthread.daemon = True
    wthread.start()

    # Initialize MQTT Client
    mqtt_client = mqtt.Client()

    # Assign callbacks
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    # Connect to MQTT Broker
    # print("Connecting to MQTT Broker...")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

    print("\n\n\n")
    # Start MQTT loop
    mqtt_client.loop_forever()
    wthread.stop()

if __name__=="__main__":
    main()