import random

class Buffer():
    def __init__(self, do_shuffle = True):
        self.buffer = []
        self.do_shuffle = do_shuffle
    def __len__(self):
        return len(self.buffer)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, size = None):
        if not size:
            size = len(self.buffer)
        return self.buffer[:size]

    def shuffle(self):
        random.shuffle(self.buffer)

    def get_mini_batch(self, batch_size = 128):
        if self.do_shuffle:
            self.shuffle()
        res = self.buffer[:batch_size]
        return res
