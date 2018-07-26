import mxnet as mx
import threading
try:
    import Queue
except:
    import queue as Queue

class DataThread(threading.Thread):
    def __init__(self, bufferIter):
        super(DataThread, self).__init__()
        self.s = bufferIter
        self.stoped = False
    def run(self):
        while not self.stoped:
            if self.s.reset_flag:
                self.s.reset_flag = False
                # clear queue
                while not self.s.buffer.empty():
                    self.s.buffer.get()
                self.s.iterator.reset()
                self.s.reset_event.set()
            else:
                try:
                    self.s.buffer.put(self.s.iterator.next())
                except StopIteration:
                    self.s.buffer.put(None)
                    self.s.reset_for_thread.wait()

class BufferIter(mx.io.DataIter):
    def __init__(self, iterator, max_buffer_size = 16):
        self.iterator = iterator
        self.buffer = Queue.Queue(maxsize = max_buffer_size)
        self.reset_event = threading.Event()
        self.reset_for_thread = threading.Event()
        self.reset_flag = False
        self.thread = DataThread(self)
        self.thread.setDaemon(True)
        self.thread.start()
        self.reset()
    def __iter__(self):
        return self
    def reset(self):
        self.reset_flag = True
        # avoid dead lock
        if not self.buffer.empty():
            self.buffer.get()
        self.reset_for_thread.set()
        self.reset_event.wait()
        self.reset_event.clear()
        self.reset_for_thread.clear()
    @property
    def provide_data(self):
        return self.iterator.provide_data
    @property
    def provide_label(self):
        return self.iterator.provide_label
    def __next__(self):
        return self.next()
    def next(self):
        e = self.buffer.get()
        if e is None:
            raise StopIteration
        return e
    def __del__(self):
        self.thread.stoped = True
        # avoid dead lock
        if not self.buffer.empty():
            self.buffer.get()
        self.reset_for_thread.set()
        self.thread.join()
