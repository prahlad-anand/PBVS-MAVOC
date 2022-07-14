import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Cautograd import *
import os
import numpy as np
import pdb
from pdb import set_trace as bp

on_zero=-1-1j
on_non_zero=+1+1j

device_ids=[0]
device = f'cuda:{device_ids[0]}'

def custom_replace(tensor, on_zero, on_non_zero):
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_r = torch.ones(10, 32,dtype = torch.complex64).to(device)
        # real = torch.ones(10, 32, dtype=torch.float32)
        # imag = torch.ones(10, 32, dtype=torch.float32)
        # z = torch.complex(real, imag)
        self.weight_r = nn.Parameter(self.fc_r, requires_grad=True)
        print('weight_r dtype',self.weight_r.dtype)
        print('weight_r shape',self.weight_r.shape)
        print('weight_r value',self.weight_r)

    def forward(self, x):
        x = F.linear(x, self.weight_r) #,dtype = torch.complex64) #, self.bias_r)
        return x

    
net = Net()
#net = nn.DataParallel( net, device_ids = [0, 1])
net = net.to(device)
#batch_size = 256
#summary(net, input_size=(batch_size, 3, 32, 32))

optimizer = optim.AdamW(net.parameters(), lr=1e-3, betas=(0.99, 0.999), weight_decay=0.1)

criterion = Complex_Loss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968,0.48215827,0.44653124), (0.24703233,0.24348505,0.26158768))
])

train_data_path = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/1_image/'
test_data_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_EO/'
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=5, shuffle=False)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
val_loader=test_loader


pytorch_total_params = sum(p.numel() for p in net.parameters())
print(pytorch_total_params)

min_valid_loss=np.inf

for epochs in range(1):
    correct=0.0
    train_loss = 0.0
    net.train()
    for inputs, target in train_loader:
        inputs = inputs.to(device,dtype=torch.complex64)
        print('input',inputs)
        target = target.to(device)
        print('Target - ', target)
        #inputs = real_to_complex(inputs)
        target = F.one_hot(target,10)
        targets = custom_replace(target.cfloat(),on_zero,on_non_zero)
        #target_real = target.real
        #target_imag = target.imag
        #targets = torch.cat((target_real,target_imag),1)
        
        complex_out = net(inputs)
        
        #input_dim =  complex_out.shape[1] // 2

        loss = criterion(complex_out,targets)
        
        out_real = complex_out.real
        targets_real = targets.real
        correct_prediction_real = torch.eq(torch.argmax(targets_real, 1), torch.argmax(out_real, 1))
        correct += correct_prediction_real.sum().item()
        print('Before Zero grad')
        #print(net.fc_r.weight.grad)
        net.zero_grad()
        print('After Zero grad')
        #print(net.fc_r.weight.grad)
        loss.backward()
        print('After backprop')
        #print(net.weight_r.weight)
        print(net.weight_r.grad)
        optimizer.step()
        train_loss +=loss.item()
        #print(net.weight_r.weight)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    
    
    net.eval()
    val_loss = 0.0
    correct_val = 0.0
    for inputs, target in val_loader:
        inputs = inputs.to(device,dtype=torch.complex64)
        target = target.to(device)
        
        #inputs = real_to_complex(inputs)
        target = F.one_hot(target,10)
        targets = custom_replace(target.cfloat(),on_zero,on_non_zero)
        
        val_out = net(inputs)

        val_out_real = val_out.real
        val_targets_real = targets.real
        correct_prediction_real_val = torch.eq(torch.argmax(val_targets_real, 1), torch.argmax(val_out_real, 1))
        correct_val += correct_prediction_real_val.sum().item()
        valid_loss = criterion(val_out,targets)
        val_loss +=valid_loss.item()
    
    if min_valid_loss > val_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        print()
        min_valid_loss = val_loss
         
        torch.save(net.state_dict(), 'best_model.pth')

    print('Training_Loss after epoch {:.2f} is {:.2f}'.format(epochs,train_loss / len(train_loader)))
    print()
    print('Validation_Loss after epoch {:.2f} is {:.2f}'.format(epochs,(val_loss / len(val_loader))))
    accuracy = correct/len(train_dataset) * 100.0
    val_acc = correct_val/1.0 *100.0
    print(f'{accuracy:.2f}% correct for train_set')
    print()
    print(f'{val_acc:.2f}% correct for val_set')
    torch.save(net.state_dict(), 'last_model.pth')

def test():
    correct = 0.0
    net = Net().to(device)
    net.load_state_dict(torch.load('best_model.pth'))
    net.eval()
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device,dtype=torch.complex64)
            target = target.to(device)
            out = net(inputs)

            target = F.one_hot(target,10)
            targets = custom_replace(target.cfloat(),on_zero,on_non_zero)
            targets_real = targets.real
            
            output_real = out.real

            correct_prediction_real = torch.eq(torch.argmax(targets_real, 1), torch.argmax(output_real, 1))
            correct += correct_prediction_real.sum().item()
        accuracy = correct/1.0 * 100.0
        print(f'{accuracy:.2f}% correct on test_set')
    return 
test()
