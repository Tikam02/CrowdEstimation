import cv2
import mxnet as mx
import numpy as np

class IMGIter(mx.io.DataIter):
    def __init__(self, batch_size, fnames, mean_value):
        super(IMGIter, self).__init__()
        self.batch_size = batch_size
        self.fnames = fnames
        self.mean_value = mean_value
        self.reset()
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def reset(self):
        self.img_iter = self.get_iter()
        self.data = None 
        self.pad = 0
        self.img_iter_over = False
        self.stop_iteration = False
        self.prepare_next()
    def get_iter(self):
        while 1:
            for fname in self.fnames:
                if self.img_iter_over:
                    self.pad += 1
                yield cv2.imread(fname).transpose((2, 0, 1)).astype(np.float32)
            self.img_iter_over = True
    @property
    def provide_data(self):
        return [('data', self.data.shape)] 
    def prepare_next(self):
        batch = [next(self.img_iter) for _ in range(self.batch_size)]
        self.data = mx.nd.array(batch) - self.mean_value
    def next(self):
        if self.stop_iteration:
            raise StopIteration
        batch = mx.io.DataBatch(data = [self.data], provide_data = self.provide_data, pad = self.pad)
        if self.img_iter_over:
            self.stop_iteration = True
        else:
            self.prepare_next()
        return batch
