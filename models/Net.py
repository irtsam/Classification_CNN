import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        """The following are the two convolutional layers and the first one 
        comprises of 1 input channel i.e. the images, 6 output channel and 
        convolutional kernel of sie 5"""
        self.conv1= nn.Conv2d(3,6,5)
        """Input channels of size 6, output channels of size 16 and kernel of
        size 5"""
        self.conv2= nn.Conv2d(6,16,5)
        #self.conv2_drop = nn.Dropout2d()
        """Fully connected neural network portion"""
        self.fc1= nn.Linear(16*5*5, 120)
        self.fc2= nn.Linear(120, 84)
        
        self.fc3= nn.Linear(84, 10)
       
        
        
        """Thus our neural network parameters have been defined"""
    def forward(self,x):
        """The x will be the output vector to feed forward the network. we 
        perform the RelU portion and feeding forward based on our input over
        here as well"""
        
        """Below is the feed forward to convolve, apply relu and Max_pool our code"""        
        x= f.max_pool2d(f.relu(self.conv1(x) ), (2,2))
        #print(x.size())
        """The next convolutional layer again maxpool size is 2*2"""
        x= f.max_pool2d(f.relu(self.conv2(x)),2)
        """Now we flatten our convolved output as a row"""
        #print(x.size())
        #x = f.dropout(x, training=self.training)
        x= x.view(-1, self.num_flat_features(x))
        """The reLU portion actions on the input depending on the object passed
        to it i.e acts both on convolutional objects and linear objects"""
        
        x= f.relu(self.fc1(x))
        #print(x.size())
        
        x= f.relu(self.fc2(x))
        #x = f.dropout(x, p=0.1,inplace= True)
        #print(x.size())
        x= self.fc3(x)
     
        
        return x
        
    def num_flat_features(self,x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        #print(size)
        num_features = 1
        for s in size:
            num_features *= s
        #print(num_features)
        return num_features
