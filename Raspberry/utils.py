import threading

class FIFOBuffer:
    def __init__(self, length):
        self.buf = []
        self.length = length
        self.condition = threading.Condition()  # Condition variable for synchronization

    def enqueue(self, data):
        with self.condition:
            self.buf.extend(data)
            self.condition.notify_all()  # Notify any waiting thread(s) that data has been added

    def dequeue(self):
        with self.condition:
            # Wait until the buffer has enough data
            while len(self.buf) < self.length:
                self.condition.wait()
            # Take the required amount of data from the buffer
            data = self.buf[:self.length]
            del self.buf[:self.length]
            print(len(self.buf))
            return data