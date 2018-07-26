import sys
sys.path.append('./vgg-mx')
import mxnet as mx
from symbol_vgg import VGG

def get_symbol():
    vgg = VGG()
    sym_vgg = vgg.get_symbol(num_classes = 1000, 
              blocks = [(2, 64),
                        (2, 128),
                        (3, 256), 
                        (3, 512)])
    relu4_3 = sym_vgg.get_internals()['relu4_3_output']
    '''
    conv6_1: 512
    conv6_2: 512
    conv6_3: 512
    conv7_1: 256
    conv7_2: 128 
    conv7_3: 64
    fu1: 1x1x1 conv
    '''
    dilated_conv_list = [
        ('conv6_1', 512),
        ('conv6_2', 512),
        ('conv6_3', 512),
        ('conv7_1', 256),
        ('conv7_2', 128),
        ('conv7_3', 64),
    ]
    x = relu4_3
    for name, num_filter in dilated_conv_list:
        x = mx.sym.Convolution(data = x, num_filter = num_filter, kernel = (3, 3), stride = (1, 1), pad = (2, 2), dilate = (2, 2), no_bias = False, name = name)
        x = mx.sym.Activation(data = x, act_type = 'relu', name = name.replace('conv', 'relu'))

    estdmap = mx.sym.Convolution(data = x, num_filter = 1, kernel = (1, 1), no_bias = False, name = 'fu1')
    return estdmap
