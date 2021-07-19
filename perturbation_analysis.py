from matplotlib import pylab as plt

from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from copy import copy, deepcopy
from torchvision import transforms

import numpy as np

# from tools import * 
# from models import *
# from get_data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)


# model_save_name = 'resnet_adadelta_last.pkl'
# path = F"/content/gdrive/My Drive/resnet_adadelta_last.pkl" 
#adadelta
adadelta = torch.load("./resnet_adadelta_best.pkl",map_location=torch.device('cpu'))
adam = torch.load("./resnet_adam_best.pkl",map_location=torch.device('cpu'))
sgd = torch.load("./resnet_sgd_best.pkl",map_location=torch.device('cpu'))

def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=True, train=True, root=".").train_data.float()
    
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

train_loader, test_loader = get_data_loaders(256, 256)
print('data is loaded')

def sp(image, amount):
      row,col = image.shape
      s_vs_p = 0.5
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      idx = np.random.choice(range(224*224), np.int(num_salt), False)
      out = out.reshape(image.size, -1)
      out[idx] = 1.0
      out = out.reshape(224,224)
      
      # Pepper mode
      num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
      idx = np.random.choice(range(224*224), np.int(num_pepper), False)
      out = out.reshape(image.size, -1)
      out[idx] = 1.0
      out = out.reshape(224,224)
      return out
  
def sp_wrapper(data, amount):
    np.random.seed(12345)
    for i in range(data.shape[0]):
        data_numpy = data[i,0,:,:].data.cpu().numpy()
        noisy_input = sp(data_numpy, amount)
        data[i,0,:,:] = torch.tensor(noisy_input).float().to(device)

    return data 

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Pertubation analysis
def perturb(model, test_loader, noise_level, ntype='white'):
    acc = []
    
    for level in tqdm(noise_level):
        correct = 0
        total_num = 0        
        
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            #print(data.min())
            #print(data.max())
            
            if ntype == 'white':
                # ******************bens stuff ******************
                # np.random.seed(123456)
                noise=torch.randn(data.shape).to(device)
                print('torch.randn(data.shape).to(device) ',noise.shape)
                print('data: ',data.shape)
                data_perturbed = data + noise * level
                print('data_perturbed ',data_perturbed.shape)
                data_perturbed = torch.clamp(data_perturbed, 0, 1)
                # ******************bens stuff above ******************

                # row,col,ch= data.shape[1:]
                # mean = 0
                # var = 0.1
                # sigma = var**0.5
                # gauss = np.random.normal(mean,sigma,(row,col,ch))
                # gauss = gauss.reshape(row,col,ch)
                # print('gauss type: ',type(gauss))
                # data_perturbed = data + gauss

            elif ntype == 'sp':
                data_perturbed = sp_wrapper(data, level)
                
            output = model(data_perturbed)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
                
        accuracy = correct / total_num
        print(accuracy)
        acc.append(accuracy)
    
    return acc   

# ntype = 'white'
ntype = 'sp'

if ntype == 'white':
    # noise_level = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    # # noise_level = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    noise_level = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]
    
else:
    noise_level = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]

acc_adadelta_sp = perturb(adadelta, test_loader, noise_level, ntype=ntype)
acc_adam_sp = perturb(adam, test_loader, noise_level, ntype=ntype)
acc_sgd_sp = perturb(sgd, test_loader, noise_level, ntype=ntype)


print(acc_adadelta_sp)
print(acc_adam_sp)
print(sgd)



plt.plot(noise_level, acc_adadelta_sp, 'h--', lw=2, c='#756bb1', label='adadelta')
plt.plot(noise_level, acc_adam_sp, 'h--', lw=2, c='#756bb1', label='adam')
plt.plot(noise_level, sgd, 'h--', lw=2, c='#e41a1c', label='sgd')
plt.xlabel('Sp perturbations', fontsize=18)
plt.ylabel('Test accuracy', fontsize=16)
plt.tick_params(axis='y', labelsize=22) 
plt.tick_params(axis='x', labelsize=22) 
plt.locator_params(axis='y', nbins=8)  
plt.legend(loc="lower left", fontsize=22)  
plt.tight_layout()
plt.show()
plt.savefig('plot_perturb.pdf')     


