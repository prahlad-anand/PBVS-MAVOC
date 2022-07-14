import albumentations as A
import cv2
import torch
import glob
import os

aug_name = 'Rotate5P0.3File600'
source_base_EO = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/aug_tests/reduced_size/' 
# '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data_er/'
# '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/17_06_augmentations/Hflip/'
# '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data/'
dest_base = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/27_06_P0.3/' + aug_name + '/'
# source_base_SAR = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/SAR_data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
# Copy files as-is with j as upper limit
for i in range(0, 10):
    source = source_base_EO + str(i)
    dest = dest_base + str(i)
    print(i)
    j=0
    for img in glob.glob(os.path.join(source, "*.png")):
        name = os.path.basename(img)
        name = name.split(".")[0]
        name = str(name)+'.png'
        n = cv2.imread(img)
        image = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dest, name), image) 
        j+=1
        if j==600:
            break
"""

t_p = {"x": 0.15, "y": 0.0}            
transform = A.Compose([
    #A.VerticalFlip(p=1.0),
    #A.HorizontalFlip(p=1.0)
    #A.RandomBrightnessContrast(p=0.2),
    A.geometric.rotate.Rotate(limit=[-5,5],p=0.3)
    #A.crops.transforms.RandomCrop(28,28,p=1.0)
    #A.geometric.transforms.Affine(scale=None, translate_percent=t_p, cval=0, mode=0, p=1.0)
])

# Outer loop is used to augment only for low-sample classes
# Inner loop performs actual augmentation according to transform function
for i in range(0, 10):
    dest = dest_base + str(i)
    source = source_base_EO + str(i)
    print(i)
    j=0
    for img in glob.glob(os.path.join(source, "*.png")):
        name = os.path.basename(img)
        name = name.split(".")[0]
        name = str(name)+'aug_'+aug_name+'.png'
        n = cv2.imread(img)
        image = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite(os.path.join(dest, name), transformed_image)
        j+=1
        if j==600:
            break
