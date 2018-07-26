import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
import mxnet as mx
import numpy as np
import cv2
import symbol_csrnet
from collections import namedtuple
from mutableModule import MutableModule
from imgiter import IMGIter
from bufferIter import BufferIter

batch_size = 1
ctx = mx.gpu(0)
MEAN_COLOR = mx.nd.array([92.8207477031, 95.2757037428, 104.877445883]).reshape((1, 3, 1, 1)) # BGR

def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

sym = symbol_csrnet.get_symbol()

mod = MutableModule(
        context = ctx,
        symbol = sym,
        data_names = ("data", ),
        label_names = ()
)

mod.bind(data_shapes = [("data", (4, 3, 512, 512))])

mod.load_params("./models/shanghaia.params")

Batch = namedtuple('Batch', ['data', 'provide_data'])

def predict(batch):
    mod.forward(batch, is_train = False)
    outputs = mod.get_outputs()[0]
    ps = outputs.sum(0, exclude = True).asnumpy()
    return ps

if __name__ == '__main__':
    import os
    def is_image_file(fname):
        ext = os.path.splitext(fname)[-1].lower()
        return ext == '.jpg'
    data_path = './data/ShanghaiTech/part_A_final/test_data/'
    image_names = list(filter(is_image_file, os.listdir(os.path.join(data_path, 'images'))))
    image_paths = [os.path.join(data_path, 'images', e) for e in image_names]

    fout = open('predict.txt', 'w')
    num_images = len(image_names)
    imgiter = IMGIter(batch_size, image_paths, MEAN_COLOR)
    # imgiter = BufferIter(imgiter, max_buffer_size = 5)
    i = 0
    for batch in imgiter:
        ps = predict(batch)
        for b in range(batch_size - batch.pad):
            img_name = image_names[i]
            p = ps[b]
            print ('{}/{}: {} {}'.format(i + 1, num_images, img_name, p))
            fout.write('{} {}\n'.format(img_name, p))
            i += 1
