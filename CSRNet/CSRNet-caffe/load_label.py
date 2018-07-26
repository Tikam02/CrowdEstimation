import scipy.io
def load_label(fname):
    data = scipy.io.loadmat(fname)
    return data['image_info'][0][0][0][0][0]

if __name__ == '__main__':
    fname = './data/ShanghaiTech/part_A_final/test_data/ground_truth/GT_IMG_1.mat'
    label = load_label(fname)
    print (label)
