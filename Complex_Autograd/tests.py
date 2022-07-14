"""
import numpy as np
import torch

dat = np.array([[1. + 2.j, 3. + 4.j], [5. + 6.j, 7. + 8.j]], dtype=complex)
x = torch.tensor(dat, requires_grad=True)

y = 2.0 * x**2

extern_grad = torch.tensor(np.ones_like(dat))
y.backward(gradient=extern_grad)

print(4 * dat)
print(x.grad)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from clean_modules import *
from torchinfo import summary

on_zero=-1-1j
on_non_zero=+1+1j

device_ids=[0]
device = f'cuda:{device_ids[0]}'

'''def one_hot(encoded):
    pos = torch.tensor([1], dtype=torch.float32)
    #neg = torch.tensor([1], dtype=torch.float32)
    neg = torch.neg(pos)
    zero_indices = encoded == 0
    non_zero_indices = encoded != 0
    encoded[non_zero_indices] = torch.complex(pos, pos)
    encoded[zero_indices] = torch.complex(neg,neg)

    return encoded'''

def custom_replace(tensor, on_zero, on_non_zero):
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

# 1. Build a computation graph
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = Cardioid()
        self.max_pool2d = max_pool()
        self.conv1 = New_ComplexConv(1,16,kernel_size=(3,3), stride=1,padding=0)
        self.conv2 = New_ComplexConv(16,32,kernel_size=(3,3), stride=1,padding=0)
        self.conv3 = New_ComplexConv(32,64,kernel_size=(3,3), stride=1,padding=0)
        self.out = New_ComplexLinear(576, 10)

        """
        Parameters
        ----------

        x = (3*3*3)+1)*16) + (3*3*16)+1)*32) + (3*3*32)+1)*64) + (10*576)+1)*10)

        """
        '''self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(36864, 10)'''

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.max_pool2d(x,(6,6))
        '''input_dim = x.shape[1] // 2
        x_real = x[:, :input_dim, :, :]
        x_imag = x[:, input_dim:, :, :]
        x = torch.complex(x_real, x_imag)'''
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.out(x)
        #output = F.log_softmax(x, dim=1)
        return x

        '''x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output'''
    
net = Net()
net = net.to(device)
batch_size = 512
#summary(net, input_size=(batch_size, 2, 28, 28))

optimizer = optim.AdamW(net.parameters(), lr=1e-3)  # 2. Setup optimizer

#criterion = nn.CrossEntropyLoss()
criterion = Complex_Loss()
#criterion = nn.MSELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.CIFAR10(
    'data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
val_dataset = datasets.CIFAR10(
    'data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512)


for epochs in range(10):
    for inputs, target in train_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        inputs = normal_img(inputs)
        #inputs = real_to_complex(inputs)
        #inputs = transform_data(inputs)
        #print(inputs.dtype)
        #print(torch.max(inputs))
        target = F.one_hot(target,10)
        target = custom_replace(target.cfloat(),on_zero,on_non_zero)
        target_real = target.real
        target_imag = target.imag
        targets = torch.cat((target_real,target_imag),1)
        #target = F.one_hot(target,10)
        #out = net(inputs)
        #output_real, output_imag = net(inputs)
        #complex_out = torch.tensor(torch.complex(output_real,output_imag), requires_grad=True)
        complex_out = net(inputs)
        #print(complex_out)
        #print(complex_out.shape)
        #inp_dim =  complex_out.shape[1] // 2
        #loss = criterion(complex_out[:,:inp_dim], target.float())
        loss = criterion(complex_out,targets)
        #loss.requires_grad= True
        #print([x.grad for x in net.parameters()])
        print(round(loss.item(), 2))
        #print(complex_out)

        net.zero_grad()
        loss.backward()
        print("Conv 1 - \n\n\n\n")
        print(net.conv1.weight.grad) 
        print(net.conv1.bias.grad)
        print("Conv 2 - \n\n\n\n")
        print(net.conv2.weight.grad) 
        print(net.conv2.bias.grad)
        print("Conv 3 - \n\n\n\n")
        print(net.conv3.weight.grad) 
        print(net.conv3.bias.grad)
        optimizer.step()

correct = 0.
net.eval()
for inputs, target in val_loader:
    inputs = inputs.to(device)
    target = target.to(device)
    inputs = normal_img(inputs)
    #inputs = fft_image(inputs)
    #inputs = real_to_complex(inputs)
    out = net(inputs)
    inp_dim =  out.shape[1] // 2

    target = F.one_hot(target,10)
    target = custom_replace(target.cfloat(),on_zero,on_non_zero)
    target_real = target.real
    target_imag = target.imag
    targets = torch.cat((target_real,target_imag),1)
    targets_real = targets[:,:inp_dim]
    #print(targets_real.shape)
    #inputs = fft_image(inputs)
    #output_real, output_imag = net(inputs)
    #out = net(inputs)
    #inp_dim =  out.shape[1] // 2

    #targets_real = (F.one_hot(target,10)).float()

    output_real = out[:,:inp_dim]

    target_abatch = targets_real[0]
    output_real_abatch = output_real[0]
    #print(output_real.shape)
    #real = output_real.real
    '''_, pred = output_real.max(1)
    #correct += (pred == target).sum()
    correct += (pred == target).sum()
accuracy = correct / len(val_dataset) * 100.
print(f'{accuracy:.2f}% correct')'''

    correct_prediction_real = torch.eq(torch.argmax(targets_real, 1), torch.argmax(output_real, 1))
    #val_loss = criterion(out,targets)
    val_loss = criterion(output_real,targets_real)
    print("Test_loss: ", val_loss)
    print("Output_abatch_real:", output_real_abatch)
    print("target_abatch_real:", target_abatch)
    correct += correct_prediction_real.sum().item()
accuracy = correct/len(val_dataset) * 100.0
print(f'{accuracy:.2f}% correct')
