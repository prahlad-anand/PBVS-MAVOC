import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import cv2
import glob
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_param = 'LeNetRotate5P0.3File'   #Just change this everytime - baseline, 55, 1010, 1515, 2020, 2525, 3030, Hflip
train_data_path='/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data_er'
#'/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/27_06_P0.3/Rotate5P0.3File600'
#'/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/17_06_augmentations/' + test_param
#'/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data_er/'
test_data_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_EO'
model_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/' + test_param + '.pth'
conf_matrix_map = test_param + '_heatmap.png'
epoch_loss_graph = test_param + '_epoch_loss_graph.png'

num_epochs = 50
batch_size = 20
num_workers = 500   # gpu-related
num_folds = 10

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

def plot(imgs, targets, name):
    fig = plt.figure(figsize=(10, 7))
    rows = 4
    columns = 5
    for i in range(len(imgs)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(str(targets[i]))
    plt.tight_layout()
    plt.savefig('/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/24_06_bs_'+str(batch_size)+'/'+test_param+'_batch'+name+'.png')


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


def train(train_data_path, model_path, epoch_loss_graph):
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=test_transform)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)   
    
    # model = LeNet()
    model = models.resnet18(pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.FocalLoss()

    optimizer = optim.Adam(model.parameters(),lr=0.003)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("Starting training on EO data")
    print()
    loss_vals = []
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
            
            """
            # Code to visualize batches
            if epoch==0 and batch_idx<2:
                img_list = []
                target_list = []
                img_count=0
                for i in range(images.shape[0]):
                    target_list.append(labels[i].numpy())
                    img_list.append(torch.permute(images[i], (1, 2, 0)).numpy())
                plot(img_list, target_list, str(batch_idx))

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
                             
        loss_vals.append(sum(epoch_losses)/len(epoch_losses))
        epoch_loss = float(n_loss/len(train_data))
        if epoch_loss<min_epoch_loss:
            min_epoch_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
        epoch_acc = n_correct.to('cpu')/total * 100.0
        print(f'Epoch ' + str(epoch+1) + ' - Loss = ' + str(epoch_loss) + ', Accuracy = ' + str(epoch_acc.numpy()) + '%')
    epoch_loss_plotter(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals, epoch_loss_graph)
      
        
def test(test_data_path, string, conf_matrix_map, model_path):
    # Conf matrix code
    y_pred_test = []
    y_true_test = []
    # cmt = torch.zeros(10, 10, dtype=torch.int64)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)   
    
    net = torch.load(model_path)
    model.load_state_dict(net)
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
    # print(cmt)

    print('Accuracy of the network on the ' + str(total) + string + ' images: %f %%' %(100 * correct / total))
    return float(100*correct/total)
           
            
def k_folds():
    num_images = []
    validation_data_path = "/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/small_aug_tests/1x5_kfold_validation/"
    subtrain_data_path = "/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/small_aug_tests/1x5_kfold_subtrain/"

    for i in range(0, 10):
        source = train_data_path + str(i)
        print(i)
        j=0
        for img in glob.glob(os.path.join(source, "*.png")):
            j+=1
        num_images.append(j)
    subtrain_acc = 0.0
    validation_acc = 0.0
    train_acc = 0.0
    test_acc = 0.0     
    for fold in range(0, num_folds): # fold - choose which set of images to set as validation
        subtrain_data_path_fold = subtrain_data_path + str(fold) + "/"
        validation_data_path_fold = validation_data_path + str(fold) + "/"
        for i in range(0, 10):  # each class
            train_data_path_subfolder = train_data_path + str(i) + "/"
            subtrain_data_path_subfolder = subtrain_data_path_fold + str(i) + "/"
            validation_data_path_subfolder = validation_data_path_fold + str(i) + "/"
            print(i)
            j=0
            for img in glob.glob(os.path.join(train_data_path_subfolder, "*.png")):
                name = os.path.basename(img)
                name = name.split(".")[0]
                name = str(name)+'.png'
                n = cv2.imread(img)
                image = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
                j+=1
                if (j >= (fold*(num_images[i])/10) and j <= (fold+1)*(num_images[i]/10)):
                    cv2.imwrite(os.path.join(validation_data_path_subfolder, name), image)
                else:
                    cv2.imwrite(os.path.join(subtrain_data_path_subfolder, name), image)
               
        train(subtrain_data_path_fold)
        subtrain_acc += test(subtrain_data_path_fold, ' subtrain', ('fold_'+str(fold)+'_subtrain_heatmap.png'))
        validation_acc += test(validation_data_path_fold, ' validation', ('fold_'+str(fold)+'_validation_heatmap.png'))
        train_acc += test(train_data_path, ' total train', ('fold_'+str(fold)+'_train_heatmap.png'))
        test_acc += test(test_data_path, ' test', ('fold_'+str(fold)+'_test_heatmap.png'))
        
    subtrain_acc/=num_folds
    validation_acc/=num_folds
    train_acc/=num_folds
    test_acc/=num_folds
    print("Average accuracy on subset = " + '%f %%' %(subtrain_acc))
    print("Average accuracy on validation set = " + '%f %%' %(validation_acc))
    print("Average accuracy on complete training set = " + '%f %%' %(train_acc))
    print("Average accuracy on test set = " + '%f %%' %(test_acc))


def temp_iter():
    lists = ['ALeft5', 'ALeft10', 'ALeft15', 'ARight5', 'ARight10', 'ARight15']
    for test_param in lists:
        model_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/' + test_param + '.pth'
        test_data_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_EO'
        train_data_path='/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/17_06_augmentations/' + test_param
        #'/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data_er/'
        epoch_loss_graph = test_param + '_epoch_loss_graph.png'
        train(train_data_path, model_path, epoch_loss_graph)
        test(test_data_path, ' test', (test_param + '_test_heatmap.png'), model_path)
        print('\n\n\nAbove was for - '+test_param+'\n\n\n')


if __name__ == "__main__":
    #temp_iter()

    # Non k fold
    train(train_data_path, model_path, epoch_loss_graph)
    test(test_data_path, ' test', conf_matrix_map, model_path)
    
    # k_folds()
