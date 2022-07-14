import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import os
from torch.nn import Parameter, init
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair
#import cv2
import pdb
from pdb import set_trace as bp

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normal_img(x):
    real = x
    imag = torch.zeros(x.shape).to(device)
    img = torch.cat((real, imag),1)
    return img

def real_to_complex(x):
    a = torch.tensor([0.4770]).to(device)
    b = torch.tensor([0.8653]).to(device)
    alpha = torch.tensor([0.4092]).to(device)
    z = (torch.sin(a*x+(1j*(b*x))+alpha)).to(device)
    real = z.real
    imag = z.imag
    z = torch.complex(real,imag)
    return z

################################## Complex BatchNorm code is taken from the DCN implementation ######################################################
class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        #print(input.shape)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0,2,3])
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps 
       
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None,:,None,None]*input.real+Rri[None,:,None,None]*input.imag).type(torch.complex64) \
                + 1j*(Rii[None,:,None,None]*input.imag+Rri[None,:,None,None]*input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
                    self.bias[None,:,0,None,None]).type(torch.complex64) \
                    +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
                    self.bias[None,:,1,None,None]).type(torch.complex64)

        return input

########################################################################################################################################

def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

def magn_phase(x):
    magn_phase = abs(x+1/x)
    return magn_phase
        
def complex_max_pooling(x,pool_size):
    complex_net = magn_phase(x)
    #maxpool = torch.nn.MaxPool2d(2,2, return_indices=True)
    maxpool = torch.nn.MaxPool2d(pool_size[0],pool_size[1], return_indices=True)
    complex_net, indices = maxpool(complex_net)
    y = retrieve_elements_from_indices(x,indices)
    y_real = y.real
    y_imag = y.imag
    y = torch.complex(y_real,y_imag)
    return y

def cardioid(x):
    phase = x.angle()
    scale = 0.5*(1.0+torch.cos(phase))
    real = x.real*scale
    imag = x.imag*scale
    x = torch.complex(real,imag)
    return x

def cardioid_linear(x):
    phase = x.angle()
    scale = 0.5*(1.0+torch.cos(phase))
    real = x.real*scale
    imag = x.imag*scale
    x = torch.complex(real,imag)
    return x

def complex_exp(x):
    x_real = x.real
    x_imag = x.imag
    exp_real = torch.exp(x_real)
    exp_imag = (torch.cos(x_imag))+(1j*(torch.sin(x_imag)))
    return (exp_real*exp_imag)

class Complex_ReLU(nn.Module):
    def forward(self,input):
         return ReLU1234(input)

class Cardioid(nn.Module):
    def forward(self,input):
         return cardioid(input)

class Cardioid_Linear(nn.Module):
    def forward(self,input):
         return cardioid_linear(input)

class Complex_ReLU14(nn.Module):
    def forward(self,input):
         return ReLU14(input)

class Exp_act(nn.Module):
    def forward(self,input):
         return complex_exp(input)

class complex_img(nn.Module):
    def forward(self,input):
         return fft_image(input)

class cRelu(nn.Module):
    def __init__(self):
        super(cRelu,self).__init__()
        self.relu = nn.ReLU()
    def forward(self,x):
        x_real = x.real
        x_imag = x.imag
        real_relu = self.relu(x_real)
        imag_relu = self.relu(x_imag)
        crelu = torch.complex(real_relu,imag_relu)
        return crelu


def complex_dropout(input, p=0.5, training=True):
    mask = torch.ones(input.shape, dtype = torch.float32).to(device)
    mask = F.dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    dropout_out = mask*input
    dropout_real = dropout_out.real
    dropout_imag = dropout_out.imag
    dropout_out = torch.complex(dropout_real,dropout_imag)
    return dropout_out

class dropout(nn.Module):
    def forward(self,input,p):
        return complex_dropout(input,p)

class max_pool(nn.Module):
    def forward(self, input, pool_size):
        return complex_max_pooling(input,pool_size)

def error_reg1(out, g_t):
    out_real = out.real
    out_imag = out.imag
    g_t_real = g_t.real
    g_t_imag = g_t.imag

    real_mul = out_real*g_t_real
    g_o_indices = real_mul >= 0.0
    out_real[g_o_indices] == 0.0
    out_imag[g_o_indices] == 0.0
    output = torch.complex(out_real,out_imag)
    g_t_real[g_o_indices] == 0.0
    g_t_imag[g_o_indices] == 0.0
    gt = torch.complex(g_t_real,g_t_imag)
    error = gt-output

    error_real = error.real
    error_imag = error.imag
    error_imag_conj = torch.neg(error_imag)
    loss_val = ((error_real*error_real)-(error_imag*error_imag_conj))
    loss_val = torch.mean(loss_val)

    return loss_val

def regularizer(out,g_t,epoch):
    input_dim = out.shape[1] // 2
    out_real = out[:,:input_dim]
    out_imag = out[:,input_dim:]
    g_t_real = g_t[:,:input_dim]
    g_t_imag = g_t[:,input_dim:]
    out = torch.complex(out_real,out_imag)
    g_t = torch.complex(g_t_real, g_t_imag)
    err = torch.tensor([0.0]).type(torch.complex64).to(device)
    e_t_init = 0.5
    if (epoch%10==0):
        e_thresh = e_t_init*np.exp(-0.2)
        e_t_init = e_thresh
    else:
        e_thresh = e_t_init
    correct_prediction = torch.eq(torch.argmax(g_t_real, 1), torch.argmax(out_real, 1))
    for i in range(len(correct_prediction)):
        e = torch.max(abs(error_reg(out[i],g_t[i])))
        if((correct_prediction[i]==True) and (e < e_thresh)):
            zero_C = torch.tensor([0.0]).type(torch.complex64).to(device)
            loss_e = error_reg(zero_C,zero_C).to(device)
        else:
            loss_e = error_reg(out[i],g_t[i]).to(device)
        err = torch.cat((err,loss_e))
    
    error_real = err.real
    error_imag = err.imag
    error_imag_conj = torch.neg(error_imag)
    loss_val = ((error_real*error_real)-(error_imag*error_imag_conj))
    loss_val = torch.mean(loss_val)
    return loss_val

def com_error(out, g_t):
    #input_dim = out.shape[1] // 2
    #out_real = out[:,:input_dim]
    #out_imag = out[:,input_dim:]
    #g_t_real = g_t[:,:input_dim]
    #g_t_imag = g_t[:,input_dim:]
    #out = torch.complex(out_real,out_imag)
    #g_t = torch.complex(g_t_real, g_t_imag)
    #out_real = out.real
    #out_imag = out.imag
    #g_t_real = g_t.real
    #g_t_imag = g_t.imag
    #real_mul = out_real*g_t_real
    #g_o_indices = real_mul >= 0.0
    #out_real[g_o_indices] == 0.0
    #out_imag[g_o_indices] == 0.0
    #output = torch.complex(out_real,out_imag)
    #g_t_real[g_o_indices] == 0.0
    #g_t_imag[g_o_indices] == 0.0
    #gt = torch.complex(g_t_real,g_t_imag)
    error = g_t-out
    error_conj=torch.conj(error)
    E=(error)*(error_conj)
    E=E.type(torch.float)

    #error_real = error.real
    #error_imag = error.imag
    #error_imag_conj = torch.neg(error_imag)
    #loss_val = ((error_real*error_real)-(error_imag*error_imag_conj))
    loss_val = torch.mean(E)

    return loss_val

def error_reg(out, g_t):
    """
    out, g_t are complex valued tensor inputs.
    Returns: Loss value(Real number) according to Hinge Error Function.
    """
    out_real = out.real
    g_t_real = g_t.real
    
    real_mul = out_real*g_t_real
    g_o_indices = real_mul >= 0.0
    g_o_indices_mask = ~g_o_indices
    
    output = out*g_o_indices_mask
    gt = g_t*g_o_indices_mask
    
    error = gt-output

    error_conj=torch.conj(error)
    E=(error)*(error_conj)
    E=E.type(torch.float)
    
    #error_real = error.real
    #error_imag = error.imag
    #error_imag_conj = torch.neg(error_imag)
    #loss_val = ((error_real*error_real)-(error_imag*error_imag_conj))
    loss_val = torch.mean(E)
    return loss_val

class Complex_Loss(nn.Module):
    def forward(self,out,g_t):
        return com_error(out, g_t)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class New_ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(New_ComplexLinear, self).__init__()
        
        self.fc_r = torch.nn.init.normal_(torch.randn(out_features, in_features)).to(device)
        self.fc_i = torch.nn.init.normal_(torch.randn(out_features, in_features)).to(device)

        self.b_r = torch.nn.init.normal_(torch.randn(out_features)).to(device)
        self.b_i = torch.nn.init.normal_(torch.randn(out_features)).to(device)

        self.weight_r = nn.Parameter(self.fc_r, requires_grad=True)
        self.weight_i = nn.Parameter(self.fc_i, requires_grad=True)

        self.bias_r = nn.Parameter(self.b_r, requires_grad=True)
        self.bias_i = nn.Parameter(self.b_i, requires_grad=True)

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        real_real = F.linear(x_real, self.weight_r, self.bias_r)
        imag_imag = F.linear(x_imag, self.weight_i, self.bias_i)
        real_imag = F.linear(x_imag, self.weight_r, self.bias_r)
        imag_real = F.linear(x_real, self.weight_i, self.bias_i)
        real = real_real - imag_imag
        imag = real_imag + imag_real
        out = torch.complex(real,imag)
        return out

class New_ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(New_ComplexConv,self).__init__()
        self.padding = padding
        self.stride = stride

        self.real_filter = torch.nn.init.xavier_normal_(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1]),gain=1.0).to(device)
        self.imag_filter = torch.nn.init.xavier_normal_(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1]),gain=1.0).to(device)
        self.C_filter = torch.complex(self.real_filter, self.imag_filter)

        '''self.mag_filter = torch.nn.init.xavier_normal_(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1]),gain=1.0).to(device)
        self.phase_filter = torch.nn.init.uniform_(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1]),a=-math.pi,b=math.pi).to(device)

        self.real_filter = self.mag_filter*torch.cos(self.phase_filter)
        self.imag_filter = self.mag_filter*torch.sin(self.phase_filter)'''

        self.weightr = nn.Parameter(self.C_filter.real, requires_grad=True)
        #self.biasr = nn.Parameter(torch.Tensor(1), requires_grad=True).to(device)
        self.weighti = nn.Parameter(self.C_filter.imag, requires_grad=True)
        #self.biasi = nn.Parameter(torch.Tensor(1), requires_grad=True).to(device)

        #self.conv_real = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        #self.conv_imag = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        real_real = F.conv2d(x_real, self.weightr, stride=self.stride, padding=self.padding)
        imag_imag = F.conv2d(x_imag, self.weighti, stride=self.stride, padding=self.padding)
        real_imag = F.conv2d(x_imag, self.weightr, stride=self.stride, padding=self.padding)
        imag_real = F.conv2d(x_real, self.weighti, stride=self.stride, padding=self.padding)
        real = real_real - imag_imag
        imag = real_imag + imag_real
        conv_map = torch.complex(real,imag)
        return conv_map

def cplx_trabelsi_standard_(weight_tensor, kind="glorot"):
    """Standard complex initialization proposed in Trabelsi et al. (2018)."""
    kind = kind.lower()
    assert kind in ("glorot", "xavier", "kaiming", "he")

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight_tensor)
    if kind == "glorot" or kind == "xavier":
        scale = 1 / math.sqrt(fan_in + fan_out)
    else:
        scale = 1 / math.sqrt(fan_in)

    # Rayleigh(\sigma / \sqrt2) x uniform[-\pi, +\pi] on p. 7
    rho = np.random.rayleigh(scale, size=weight_tensor.shape)
    theta = np.random.uniform(-np.pi, +np.pi, size=weight_tensor.shape)

    # eq. (8) on p. 6
    with torch.no_grad():
        weight_tensor.real.copy_(torch.from_numpy(np.cos(theta) * rho))
        weight_tensor.imag.copy_(torch.from_numpy(np.sin(theta) * rho))

    return weight_tensor


class c_avgpool(nn.Module):
    def __init__(self, pool_size):
        super(c_avgpool,self).__init__()
        self.avg_pool = nn.AvgPool2d((pool_size[0],pool_size[1]))
    def forward(self,x):
        x_real = x.real
        x_imag = x.imag
        real_relu = self.avg_pool(x_real)
        imag_relu = self.avg_pool(x_imag)
        out = torch.complex(real_relu,imag_relu)
        return out