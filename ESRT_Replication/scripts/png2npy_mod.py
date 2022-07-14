import os
import argparse
import skimage.io as sio
import numpy as np
import pdb
import glob

pathFrom  = '/home/airl-gpu4/Aniruddh/PBVS_Challenge/Datasets/train_images/train_images/EO_data_er_imbalanced/'
pathTo  = '/home/airl-gpu4/Aniruddh/PBVS_Challenge/Datasets/train_images/train_images/EO_data_er_imbalanced_npy/'


for i in range(0, 10):
    dest = pathTo + str(i)
    source = pathFrom + str(i)
    print(i)
    n=0
    for img in glob.glob(os.path.join(source, "*.png")):
        name = os.path.basename(img)
        name = name.split(".")[0]
        np.save(os.path.join(dest, name + '.npy'), img)
        n += 1
    print('Converted ' + str(n) + ' images.')