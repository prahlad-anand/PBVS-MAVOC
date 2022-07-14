import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from torchsummary import summary
from Cautograd import *
import os
import numpy as np
import pdb
from pdb import set_trace as bp
#from einops import rearrange
#from torchinfo import summary

on_zero=-1-1j
on_non_zero=+1+1j

#device_ids=[0]
#device = f'cuda:{device_ids[0]}'

def custom_replace(tensor, on_zero, on_non_zero):
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_r = torch.ones(10, 32,dtype = torch.float32).to(device)
        real = torch.ones(10, 32, dtype=torch.float32)
        imag = torch.ones(10, 32, dtype=torch.float32)
        z = torch.complex(real, imag)
        self.weight_r = nn.Parameter(self.fc_r, requires_grad=True)
        print('weight_r dtype',self.weight_r.dtype)
        print('weight_r shape',self.weight_r.shape)
        print('weight_r value',self.weight_r)
        
        """
        self.l1 = nn.Linear(32, 10, bias = False)
        w = torch.ones(32, 10)
        nn.init.uniform_(w)
        """
        """
        self.relu = nn.ReLU()
        #self.linear_relu = Cardioid_Linear()
        self.avg_pool = nn.AvgPool2d(2, stride=1)
        
        #self.conv1 = torch.nn.Conv2d(3,16,kernel_size=(3,3),stride=2,padding=1) 
        self.conv1 = torch.nn.Conv2d(3,32,kernel_size=(3,3),stride=2,padding=1,bias=False)#,dtype=torch.complex64)                      

        #self.conv2 = torch.nn.Conv2d(16,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1,bias=False)#,dtype=torch.complex64)                     

        #self.conv3 = torch.nn.Conv2d(32,64,kernel_size=(3,3),stride=2,padding=1) 
        self.conv3 = torch.nn.Conv2d(32,64,kernel_size=(3,3),stride=2,padding=1,bias=False)#,dtype=torch.complex64)
        #self.C_Attn =  CosformerAttention(embed_dim=16,num_heads=1)
        
        self.depth_conv = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4,stride=2,bias=False, groups=64)
        #self.point_conv = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.fc1 =torch.nn.Linear(64, 10,bias=False)    ###remove this layer.
        #self.fc2 =torch.nn.Linear(128, 10)  
        #self.out = torch.nn.Linear(288,10) #.to(torch.complex64)

        #self.C_Attn =  CosformerAttention(embed_dim=16,num_heads=1)
        #self.conv4 = ComplexConv(128,128,kernel_size=3,stride=1,padding='same')
        #self.conv5 = ComplexConv(128,256,kernel_size=3,stride=1,padding=0)
        
        #self.bn1 = ComplexBatchNorm2d(32)                                                          
        #self.bn2 = ComplexBatchNorm2d(64)
        #self.bn3 = ComplexBatchNorm2d(128)
        #self.bn4 = ComplexBatchNorm2d(256)
        #self.fc1 = torch.nn.Linear(256, 128).to(torch.complex64)  
        """                                                                   

    def forward(self, x):
        x = F.linear(x, self.weight_r)
        """
        #x = self.complex_img(x)
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        
        '''amp, phase = rtheta(x)
        amp_patch = rearrange(amp, 'b c h w -> (h w) b c')
        phase_patch = rearrange(phase, 'b c h w -> (h w) b c')
        attn_out = self.C_Attn(query=amp_patch, key=phase_patch, value=phase_patch)
        attn_simi_map = rearrange(attn_out, '(h1 w1) b c -> b c h1 w1',h1=8,w1=8)
        
        x = self.conv2(attn_simi_map)
        #x = self.bn3(x)
        x = self.relu(x)'''

        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.relu(x)
        #x = self.avg_pool(x)
        x = self.depth_conv(x)
        #x = self.point_conv(x)
        #print('after pooling',x.shape)
        
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #x = self.out(x)
        #x = self.out(x)
        x = self.fc1(x)
        
        #x = self.fc2(x)
        """
        #x = self.l1(x)
        return x

    
net = Net()
#net = net.to(device)
#net = nn.DataParallel( net, device_ids = [0, 1])               #Not necessry for 1-NN
net = net.to(device)
#batch_size = 256
#summary(net, input_size=(batch_size, 3, 32, 32))

optimizer = optim.AdamW(net.parameters(), lr=1e-3, betas=(0.99, 0.999), weight_decay=0.1)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

#criterion = nn.MSELoss()
# Old - criterion = nn.CrossEntropyLoss()

# New - 
criterion = Complex_Loss()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968,0.48215827,0.44653124), (0.24703233,0.24348505,0.26158768))
])

"""
# Old stuff
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968,0.48215827,0.44653124), (0.24703233,0.24348505,0.26158768))
])
train_dataset = datasets.CIFAR10(
    'dataset', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256)

test_dataset = datasets.CIFAR10(
    'dataset', train=False, download=True, transform=transform)
test_set, val_set = torch.utils.data.random_split(test_dataset, [7000, 3000])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=256)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256)
"""

# New
train_data_path = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/1_image/'
test_data_path = '/home/airl-gpu4/Aniruddh/PBVS_repro/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_EO/'
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=5, shuffle=False)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
val_loader=test_loader

pytorch_total_params = sum(p.numel() for p in net.parameters())
print('pytorch_total_params',pytorch_total_params)

min_valid_loss=np.inf

for epochs in range(1):
    correct=0.0
    train_loss = 0.0
    net.train()
    for inputs, target in train_loader:
        inputs = inputs.to(device)
        print('input',inputs)
        target = target.to(device)
        #inputs = real_to_complex(inputs)
        target = F.one_hot(target,10)
        target = target.type(torch.float)
        #targets = custom_replace(target.cfloat(),on_zero,on_non_zero)
        #target_real = target.real
        #target_imag = target.imag
        #targets = torch.cat((target_real,target_imag),1)
        
        complex_out = net(inputs)
        
        #input_dim =  complex_out.shape[1] // 2

        loss = criterion(complex_out,target)
        
        #out_real = complex_out.real
        #targets_real = targets.real
        correct_prediction_real = torch.eq(torch.argmax(target, 1), torch.argmax(complex_out, 1))
        correct += correct_prediction_real.sum().item()
        print('Before Zero grad')
        #print(net.l1.weight.grad)
        net.zero_grad()
        print('After Zero grad')
        #print(net.l1.weight.grad)
        loss.backward()
        print('After backprop')
        #print(net.l1.weight.grad)
        optimizer.step()
        train_loss +=loss.item()
    for name, param in net.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    
        #########################################################################################Validation######################################
    net.eval()
    val_loss = 0.0
    correct_val = 0.0
    for inputs, target in val_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        #inputs = real_to_complex(inputs)
        target = F.one_hot(target,10)
        target = target.type(torch.float)
        #targets = custom_replace(target.cfloat(),on_zero,on_non_zero)
        
        val_out = net(inputs)

        #val_out_real = val_out.real
        #val_targets_real = targets.real
        correct_prediction_real_val = torch.eq(torch.argmax(target, 1), torch.argmax(val_out, 1))
        correct_val += correct_prediction_real_val.sum().item()
        valid_loss = criterion(val_out,target)
        val_loss +=valid_loss.item()
    
    if min_valid_loss > val_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        print()
        min_valid_loss = val_loss
         
        torch.save(net.state_dict(), 'best_model.pth')

    print('Training_Loss after epoch {:.2f} is {:.2f}'.format(epochs,train_loss / len(train_loader)))
    print()
    print('Real network')
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
    net.load_state_dict(torch.load('best_model.pth'), strict = False)
    net.eval()
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            #inputs = real_to_complex(inputs)
            out = net(inputs)
            #inp_dim =  out.shape[1] // 2

            target = F.one_hot(target,10)
            target = target.type(torch.float)
            #targets = custom_replace(target.cfloat(),on_zero,on_non_zero)
            #targets_real = targets.real
            
            #output_real = out.real

            correct_prediction_real = torch.eq(torch.argmax(target, 1), torch.argmax(out, 1))
            correct += correct_prediction_real.sum().item()
        accuracy = correct/1.0 * 100.0
        print(f'{accuracy:.2f}% correct on test_set')
        print('Real network')
    return 
test()
