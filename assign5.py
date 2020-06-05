# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:31:03 2020

@author: joshu
"""
import os
import json
import time
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from torch import argmax
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10



#class ImageNet(data.Dataset):
#    def __init__(self, path):
#        self.path = path
#        images = self.unpickle(self.path)
#        self.lbls = images[b'labels']
#        meta = self.unpickle('./cifar-10-batches-py/batches.meta')
#        self.lbl_dic = meta[b'label_names']
#
#        self.img_transforms = transforms.Compose([
#            transforms.ToTensor(),
#        ])
#
#        self.imgs = images[b'data']
#        
#        pass
#    def __getitem__(self,index):
#        img = self.imgs[index]
#        full_img = Image.fromarray(np.reshape(img,(32,32,3))).convert('RGB')
#        plt.imshow(np.transpose(np.reshape(img,(3,32,32))))
#        plt.show()
#        plt.close()
#        final_img = self.img_transforms(full_img)
#        lbl = self.lbls[index]
#        return final_img,lbl
#    
#    def unpickle(self,file):
#        import pickle
#        with open(file, 'rb') as fo:
#            dict = pickle.load(fo, encoding='bytes')
#        return dict
#    
#    def __len__(self):
#        return len(self.imgs)
#    
    
    
"""
Notes: lbls, taken from image name to determine what category it falls into,
image is stored in x,
and lbl of image is stored in x , to store for verification
Ask about labels in class today
how to use gpu

"""
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #create a headless ResNet
        self.conv1 = nn.Conv2d(3, 64, 5, 3)
        self.batch = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128, 3, padding=1)
        self.maxpool = nn.MaxPool2d((3,3))
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1)
        self.batch2 = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(3)
        self.batch3 = nn.BatchNorm2d(256)
        self.linear = nn.Linear(256,10)

        
        
        
        

    def forward(self, x):
        
        #pass input to headless ResNet 
        # batch_size = # of images, channels(rgb), x, y
        x = self.batch(F.relu(self.conv1(x)))
        x = self.maxpool(self.batch2(F.relu(self.conv2(x))))
        x = self.avgpool(self.batch3(F.relu(self.conv3(x))))
        x = x.view(-1,256)
        return self.linear(x)
        
    
    
def imshow(img):
   # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.close()
    
    
if __name__ == "__main__":
    
    #use cuda if available

    #create a dataset
    train_dataset = CIFAR10('./cifar-10-batches-py',train=True,download=True, transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465],[0.247, 0.243, 0.261])]))
    
    
    #Split the data into training and validation
    #create a test dataset
    #train_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=4)
    train_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=4)
    dataiter = iter(train_dataloader)
    images, labels = dataiter.next()
    # show images
    imshow(utils.make_grid(images))
    print(labels)
    
    val_dataset = CIFAR10('./cifar-10-batches-py',train=False,download=True, transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465],[0.247, 0.243, 0.261])]))
    val_dataloader = DataLoader(val_dataset, batch_size=1,shuffle=True, num_workers=4)
        
    #define model, loss function, optimizer
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    # look up later: crossentropyloss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())
    
    for epoch in range(50):
        #trainining
        t = time.time()
        
        #set modedl to train model
        model.train()
        running_loss = 0.0
        for batch_idex, (imgs, lbls) in enumerate(train_dataloader):
            # If you are using gpu, move data to gpu
            imgs = imgs.to(device)
            lbls = lbls.to(device)
           #zero paramaeter gradients
            optimizer.zero_grad()
            #get loss and do backprop
            output = model(imgs)
            loss = loss_fn(output, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            if batch_idex==len(train_dataloader)-1:
                print("Epoch {} train: {}/{} loss: {:.5f} ({:3f}s)".format(
                        epoch+1, batch_idex+1, len(train_dataloader), running_loss/len(train_dataloader), time.time()-t), end="\r")
                running_loss = 0.0
        
    
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(val_dataloader):
                
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                output = model(imgs)
                predicted = argmax(output.data)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()       
          
            print("This much, {}% , of the dataset was correct".format((correct/total)*100))
        
        
        pass




#what does optimizer do?
#asak about net module1