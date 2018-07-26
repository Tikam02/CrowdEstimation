from load_label import load_label
import os
import numpy as np

data_path = './data/ShanghaiTech/part_A_final/test_data/'
fin = open('./predict.txt')
errs = [] 
for line in fin:
    sp = line.split(' ')
    pid = int(sp[0].split('IMG_')[1].split('.jpg')[0])
    pred = float(sp[1])
    label_fname = os.path.join(data_path, 'ground_truth', 'GT_IMG_%d' % pid)
    label = load_label(label_fname)
    count = label.shape[0]
    errs.append(pred - count)

mae = np.mean(np.abs(errs))
print ("MAE: %.3f" % mae)

mse = np.sqrt(np.mean(np.square(errs)))
print ("MSE: %.3f" % mse)
