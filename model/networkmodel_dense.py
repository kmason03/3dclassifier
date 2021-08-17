# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append("/home/kmason/SparseConvNet")
import sparseconvnet as scn

import time
import math
import numpy as np
from BasicBlocks import *

#This is the dense version of the network. Won't actually work with my inputs,
# provides reference to the dense implementation.

class Classifier3d_dense(nn.Module):
    def __init__(self, inputshape3D, inputshape2D, reps, nin_features, nout_features, nplanes,show_sizes):
        nn.Module.__init__(self)
        """
        inputs
        ------
        inputshape3D [list of int]: dimensions of the 3d voxel
        inputshape2D [list of int]: dimensions of the 2d image
        reps [int]: number of residual modules per layer (for both encoder and decoder)
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that feed into the regression layer
        nPlanes [int]: the depth of the U-Net
        show_sizes [bool]: if True, print sizes while running forward
        """
        # classes: p,mu,e,pi,gamma,other(badreco)
        self._nclasses = 6
        self._nchannelsv1 = 16
        self._nchannelsv2_1 = 20
        self._nchannelsv2_2 = 30
        self._nchannelsmv = 96
        self._inputshape3d = inputshape3D
        self._show_sizes = show_sizes

        if len(self._inputshape3d)!=3:
            raise ValueError("expected inputshape3d to contain size of 3 dimensions only."
                             +"given %d values"%(len(self._inputshape3d)))
        self._inputshape2d = inputshape2D
        if len(self._inputshape2d)!=2:
            raise ValueError("expected inputshape2 to contain size of 2 dimensions only."
                             +"given %d values"%(len(self._inputshape2d)))
        # first define VCNN1
        self.VFC1 =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4000752 ,2048),
            torch.nn.Linear(2048,self._nclasses),
            torch.nn.Softmax(1)
        )

        self.VCNN1 = torch.nn.Sequential(
            torch.nn.Conv3d(1,self._nchannelsv1 , kernel_size = 3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2,stride=2),
            torch.nn.Conv3d(self._nchannelsv1,self._nchannelsv1 , kernel_size = 3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(self._nchannelsv1,self._nchannelsv1 , kernel_size = 3, stride=1, padding=0),
            torch.nn.MaxPool3d(2,stride=4),
            torch.nn.Dropout(0.5),
        )

        #VCNN2 Layers
        self.v2_conv1_1 = torch.nn.Conv3d(1,self._nchannelsv2_1 , kernel_size = 1, stride=1, padding=0)
        self.v2_conv1_2 = torch.nn.Conv3d(60,self._nchannelsv2_2 , kernel_size = 1, stride=1, padding=0)
        self.v2_conv3_1 = torch.nn.Conv3d(1,self._nchannelsv2_1 , kernel_size = 3, stride=1, padding=1)
        self.v2_conv3_2 = torch.nn.Conv3d(60,self._nchannelsv2_2 , kernel_size = 3, stride=1, padding=1)
        self.v2_conv5_1 = torch.nn.Conv3d(1,self._nchannelsv2_1 , kernel_size = 5, stride=1, padding=2)
        self.v2_Relu = torch.nn.ReLU()
        self.v2_drop_1 = torch.nn.Dropout(0.2)
        self.v2_drop_2 = torch.nn.Dropout(0.3)
        self.v2_drop_3 = torch.nn.Dropout(0.5)

        self.VFC2 =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear( 810000,2048),
            torch.nn.Linear(2048,self._nclasses),
            torch.nn.Softmax(1)
        )

        #MVCNN Layers
        self.alexnet = torch.nn.Sequential(
            torch.nn.Conv2d(1,self._nchannelsmv , kernel_size = 5, stride=1, padding=0),
            torch.nn.Conv2d(self._nchannelsmv ,256, kernel_size = 3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2),
            torch.nn.Conv2d(256 ,384, kernel_size = 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2),
            torch.nn.Conv2d(384 ,384, kernel_size = 3, stride=1, padding=1),
            torch.nn.Conv2d(384 ,256, kernel_size = 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2),
            torch.nn.Flatten()
        )

        self.MVCNNFC =torch.nn.Sequential(
            torch.nn.Linear( 6912,2304),
            torch.nn.Linear( 2304,1152),
            torch.nn.Linear(1152,self._nclasses)
            # torch.nn.Softmax(1)

        )


    def forward(self,input3d_t,input2dU_t,input2dV_t,input2dY_t,batchsize):
        print("show sizes", self._show_sizes)
        if self._show_sizes:
            print( "input3d_t ",input3d_t.shape)
            print( "input2dU_t ",input2dU_t.shape)
            print( "input2dV_t ",input2dV_t.shape)
            print( "input2dY_t ",input2dY_t.shape)

        #first v1
        xv1=input3d_t

        xv1=self.VCNN1(xv1)
        if self._show_sizes:
            print ("VCNN1: ",xv1.shape)
        xv1=self.VFC1(xv1)
        if self._show_sizes:
            print ("VCNN1: ",xv1.shape)

        # next v2
        xv2 = input3d_t
        xv2_1 = self.v2_conv1_1(xv2)
        xv2_3 = self.v2_conv3_1(xv2)
        xv2_5 = self.v2_conv5_1(xv2)

        xv2 = torch.cat((xv2_1,xv2_3,xv2_5),1)
        xv2 = self.v2_Relu(xv2)
        xv2 = self.v2_drop_1(xv2)
        if self._show_sizes:
            print("VCNN2: ", xv2.shape)

        xv2_1 = self.v2_conv1_2(xv2)
        xv2_3 = self.v2_conv3_2(xv2)

        xv2 = torch.cat((xv2_1,xv2_3),1)
        xv2 = self.v2_Relu(xv2)
        xv2 = self.v2_drop_2(xv2)
        if self._show_sizes:
            print("VCNN2: ", xv2.shape)

        xv2 = self.v2_conv3_2(xv2)
        xv2 = self.v2_Relu(xv2)
        xv2 = self.v2_drop_3(xv2)
        if self._show_sizes:
            print("VCNN2: ", xv2.shape)
        xv2 = self.VFC2(xv2)

        if self._show_sizes:
            print("VCNN2: ", xv2.shape)

        xmvU = input2dU_t
        xmvU = self.alexnet(xmvU)
        if self._show_sizes:
            print("MV-CNN - U: ", xmvU.shape)

        xmvV = input2dV_t
        xmvV = self.alexnet(xmvV)
        if self._show_sizes:
            print("MV-CNN - V: ", xmvV.shape)

        xmvY = input2dY_t
        xmvY = self.alexnet(xmvY)
        if self._show_sizes:
            print("MV-CNN - Y: ", xmvY.shape)

        xmv=torch.cat((xmvU,xmvV,xmvY),1)
        if self._show_sizes:
            print("MV-CNN - cat: ", xmv.shape)

        xmv = self.MVCNNFC(xmv)
        if self._show_sizes:
            print("MV-CNN: ", xmv.shape)

        x = torch.cat((xv1,xv2,xmv),0)
        if self._show_sizes:
            print("Final Sizes: ", x.shape)
            print(x)
        return x
