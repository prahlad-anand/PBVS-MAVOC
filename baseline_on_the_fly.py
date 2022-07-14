import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os
from collections import OrderedDict
from itertools import cycle
import cv2
import pdb
from pdb import set_trace as bp
from torchsummary import summary
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_param = 'ImbalancedLeNetRotate25LDAMLossShuffleTrue'
train_data_path = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data_er_imbalanced'
#'/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/aug_tests/reduced_size'
#'/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data_er'
test_data_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_EO/'
model_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/' + test_param + '.pth'
#'EO_cross_domain_resnet50.pth'
#'EO_focal_loss_swd_resnet50.pth'
#'unicornmodel_DA_SAR.pth'
conf_matrix_map = test_param + '_heatmap.png'
# epoch_loss_graph = test_param + '_epoch_loss_graph.png'

num_epochs = 50
batch_size = 20
num_workers = 500   # gpu-related
num_folds = 10

torch.autograd.set_detect_anomaly(True)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomRotation((-25, 25))]), p=0.3)
        ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))
# transforms.RandomRotation((-15,15))
# transforms.RandomHorizontalFlip(1.0)
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),


# Code to create batch image tiles - 
from PIL import Image
from pathlib import Path
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["savefig.bbox"] = 'tight'

def plot(imgs, targets, batch_num):
    fig = plt.figure(figsize=(10, 7))
    rows = 4
    columns = 5
    for i in range(len(imgs)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(str(targets[i]))
    plt.tight_layout()
    plt.savefig('/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/'+test_param+'_batch'+batch_num+'.png')


def epoch_loss_plotter(epochs, loss, epoch_loss_graph):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, loss)
    plt.grid()
    plt.show()
    plt.savefig(epoch_loss_graph)


class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, (5,5))
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(44944, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = x.view(x.size(0),-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

"""
model = models.resnet50(pretrained=True)
num_ftrs_EO = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs_EO, 10))
"""
model = LeNet()
"""
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
"""
model = model.to(device)


def get_cls_num_list(self):
    cls_num_list = []
    for i in range(self.cls_num):
        cls_num_list.append(self.num_per_cls_dict[i])
    return cls_num_list

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=(600.*torch.ones(1, 10)).numpy(), max_m=0.5, weight=torch.Tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(device), s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=torch.Tensor([1./600., 1./525., 1./500., 1./475., 1./450., 1./300., 1./200., 1./150., 1./200., 1./150.]).to(device), gamma=2.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


def train():
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    
    y_train_indices = len(train_data)

    y_train = [train_data.targets[i] for i in range(y_train_indices)]

    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), 293772, replacement=True)
    
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=5, shuffle=True)#,sampler = sampler)

    #, sampler = sampler

    criterion = LDAMLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting training on EO data")
    print()
    # loss_vals = []
    min_epoch_loss = 5000.0
     
    for epoch in range(num_epochs):
        epoch_losses = []
        n_loss = 0.0
        n_correct = 0 
        total = 0.0
        batch_count = 0
        # loads a batch of images
        for batch_idx, data in tqdm(enumerate(train_data_loader)): 
            # get inputs
            images, labels = data[0], data[1]
            
            
            # Code to visualize batches
            if epoch==0 and batch_idx<2:
                img_list = []
                target_list = []
                img_count=0
                for i in range(images.shape[0]):
                    target_list.append(labels[i].numpy())
                    img_list.append(torch.permute(images[i], (1, 2, 0)).numpy())
                plot(img_list, target_list, str(batch_idx))

            """
            # Pause at every batch on the first epoch after creating an image tile
            if epoch==0:
                print('Continue?')
                random_input = input()
            """

            images, labels = images.to(device), labels.to(device)                       
                  
            # forward pass
            outputs = model(images)
            outputs = outputs.to(device)
            _, preds = torch.max(outputs, 1)
        
            # compute loss
            loss = criterion(outputs, labels)          
            
            # zeroing gradients
            optimizer.zero_grad()
            
            # backprop
            loss.backward()
            epoch_losses.append(loss.item())
           
            # Optimization
            optimizer.step()
            
            n_loss += loss.item() * images.size(0)
            n_correct += torch.sum(preds == labels.data)
            total += labels.size(0)
                             
        # loss_vals.append(sum(epoch_losses)/len(epoch_losses))
        epoch_loss = float(n_loss/len(train_data))
        if epoch_loss<min_epoch_loss:
            min_epoch_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
        epoch_acc = n_correct.to('cpu')/total * 100.0
        print(f'Epoch ' + str(epoch+1) + ' - Loss = ' + str(epoch_loss) + ', Accuracy = ' + str(epoch_acc.numpy()) + '%')
    # epoch_loss_plotter(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals, epoch_loss_graph)

def test():
    # Conf matrix code
    y_pred_test = []
    y_true_test = []
    #cmt = torch.zeros(10, 10, dtype=torch.int64)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)

    net = torch.load(model_path)
    model.load_state_dict(net)
    # model.load_state_dict(copy.deepcopy(torch.load(model_path, device)))
    model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_data_loader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            
            # Conf matrix code
            target = labels
            target_test = target.data.cpu().numpy() # Without one-hot-confusion matrix
            y_true_test.extend(target_test) # Save Truth 
            complex_out_test = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred_test.extend(complex_out_test) # Save Prediction 
            target = F.one_hot(target,10)
            target = target.type(torch.float)
            """
            for i, j in zip(labels, predicted):
                cmt[i.item()][j.item()]+=1
            """
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Build confusion matrix
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    cf_matrix = confusion_matrix(y_true_test, y_pred_test)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(conf_matrix_map)
    #print(cmt)

    print('Accuracy of the network on the ' + str(total) + ' images: %f %%' %(100 * correct / total))
    return float(100*correct/total)
           

if __name__ == "__main__":
    train()
    test()
    print('\n\n\nAbove was for - '+test_param+'\n\n\n')
