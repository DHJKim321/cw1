import os, sys
sys.path.append(os.path.join(os.path.dirname(os.curdir)))

class RequestQueue:
    def __init__(self, max_queue_size: int = 100):
        self.queue = []
        self.max_queue_size = max_queue_size

    def enqueue(self, item):
        if len(self.queue) < self.max_queue_size:
            self.queue.append(item)
        else:
            raise Exception("Queue is full")

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            raise Exception("Queue is empty")

    def __len__(self):
        return len(self.queue)