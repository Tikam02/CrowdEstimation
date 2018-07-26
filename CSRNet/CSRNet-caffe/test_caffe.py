import caffe
import numpy as np
import cv2

MEAN_COLOR = np.array([92.8207477031, 95.2757037428, 104.877445883]).reshape((1, 3, 1, 1)) # BGR

deploy = './models/deploy.prototxt'
caffe_model = './models/shanghaia.caffemodel'

nn = caffe.Net(deploy, caffe_model, caffe.TEST) 

def predict(imgs):
    nn.blobs["data"].reshape(*(imgs.shape))
    nn.blobs["data"].data[...] = imgs
    nn.forward()
    ps = nn.blobs["estdmap"].data.sum((1,2,3))
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
    for i, fname in enumerate(image_paths):
        img_name = image_names[i]
        im = cv2.imread(fname)
        rows, cols, ts = im.shape
        imgs = im.transpose((2,0,1)).reshape((1, 3, rows, cols)) - MEAN_COLOR
        ps = predict(imgs)
        p = ps[0]
        print ('{}/{}: {} {}'.format(i + 1, num_images, img_name, p))
        fout.write('{} {}\n'.format(img_name, p))
        i += 1
