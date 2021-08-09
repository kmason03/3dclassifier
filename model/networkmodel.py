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

# sparse version of the network to handle our inputs.

class Classifier3d(nn.Module):
    def __init__(self, inputshape3D, inputshape2D, reps, nin_features, nout_features, nplanes,
            batchsize,show_sizes, timing):
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
        self._nchannelsv2_1 = 16
        self._nchannelsv2_2 = 24
        self._nchannelsmv = 96
        self._inputshape3d = inputshape3D
        self._show_sizes = show_sizes
        self._timing = timing
        self._batchsize = batchsize

        if len(self._inputshape3d)!=3:
            raise ValueError("expected inputshape3d to contain size of 3 dimensions only."
                             +"given %d values"%(len(self._inputshape3d)))
        self._inputshape2d = inputshape2D
        if len(self._inputshape2d)!=2:
            raise ValueError("expected inputshape2 to contain size of 2 dimensions only."
                             +"given %d values"%(len(self._inputshape2d)))

        #first change to sparse
        self._densetosparse3d = scn.DenseToSparse(3)
        self._densetosparse2d = scn.DenseToSparse(2)
        self._sparsetodense3dVCNN1 = scn.SparseToDense(3,self._nchannelsv1)
        self._sparsetodense3dVCNN2 = scn.SparseToDense(3,24)
        self._sparsetodense2d = scn.SparseToDense(2,256)

        self._softmax=torch.nn.Softmax(1)

        # first define VCNN1
        self.VFC1 =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self._batchsize*524288 ,2048),
            torch.nn.Linear(2048,self._nclasses),
            torch.nn.Softmax(1)
        )
        self.VFC1_QUAL =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self._batchsize*524288 ,2048),
            torch.nn.Linear(2048,1),
            torch.nn.Softmax(1)
        )


        self.VCNN1 = torch.nn.Sequential(
            scn.SubmanifoldConvolution(3, 1, self._nchannelsv1, 3,0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2),
            scn.SubmanifoldConvolution(3, self._nchannelsv1, self._nchannelsv1, 3,0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2),
            scn.SubmanifoldConvolution(3, self._nchannelsv1, self._nchannelsv1, 3,0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2),
            scn.SubmanifoldConvolution(3, self._nchannelsv1, self._nchannelsv1, 3,0),
            scn.ReLU(),
            scn.MaxPooling(3,4,4),
            scn.Dropout(0.5)
        )

        # #VCNN2 Layers
        self.v2_conv1_1 = torch.nn.Sequential(
            scn.SubmanifoldConvolution(3, 1, self._nchannelsv2_1, 1, 0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2)
        )
        self.v2_conv1_2 = torch.nn.Sequential(
            scn.SubmanifoldConvolution(3,self._nchannelsv2_1*3, self._nchannelsv2_2, 1, 0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2)
        )
        self.v2_conv3_1 = torch.nn.Sequential(
            scn.SubmanifoldConvolution(3, 1, self._nchannelsv2_1 , 3, 0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2)
        )
        self.v2_conv3_2 = torch.nn.Sequential(
            scn.SubmanifoldConvolution(3,self._nchannelsv2_1*3, self._nchannelsv2_2, 3, 0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2)
        )
        self.v2_conv5_1 = torch.nn.Sequential(
            scn.SubmanifoldConvolution(3, 1, self._nchannelsv2_1 , 5, 0),
            scn.ReLU(),
            scn.MaxPooling(3,2,2)
        )
        self.v2_join = scn.JoinTable()
        self.v2_Relu = scn.ReLU()
        self.v2_drop_1 = scn.Dropout(0.2)
        self.v2_drop_2 = scn.Dropout(0.3)
        self.v2_drop_3 = scn.Dropout(0.5)
        self.v2_finalmaxpool = scn.MaxPooling(3,4,4)

        self.VFC2 =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear( self._batchsize*786432  ,2048),
            torch.nn.Linear(2048,self._nclasses),
            torch.nn.Softmax(1)
        )
        self.VFC2_QUAL =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear( self._batchsize*786432  ,2048),
            torch.nn.Linear(2048,1),
            torch.nn.Softmax(1)
        )

        #MVCNN Layers
        self.alexnet = torch.nn.Sequential(
            scn.SubmanifoldConvolution(2, 1, self._nchannelsmv , 5, 0),
            scn.SubmanifoldConvolution(2, self._nchannelsmv ,256 , 3, 0),
            scn.ReLU(),
            scn.MaxPooling(2,2,2),
            scn.SubmanifoldConvolution(2, 256 ,384 , 3, 0),
            scn.ReLU(),
            scn.MaxPooling(2,4,4),
            scn.SubmanifoldConvolution(2, 384,384 , 3, 0),
            scn.SubmanifoldConvolution(2, 384, 256 , 3, 0),
            scn.ReLU(),
            scn.MaxPooling(2,4,4)
        )

        self.MVCNNFC =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear( self._batchsize*196608,2304),
            torch.nn.Linear( 2304,1152),
            torch.nn.Linear(1152,self._nclasses),
            torch.nn.Softmax(1)
        )

        self.MVCNNFC_QUAL =torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear( self._batchsize*196608,2304),
            torch.nn.Linear( 2304,1152),
            torch.nn.Linear(1152,1),
            torch.nn.Softmax(1)
        )



    def forward(self,input3d_t,input2dU_t,input2dV_t,input2dY_t):
        print("show sizes", self._show_sizes)
        if self._timing:
            time1 = time.perf_counter()
        if self._show_sizes:
            print( "input3d_t ",input3d_t.shape)
            print( "input2dU_t ",input2dU_t.shape)
            print( "input2dV_t ",input2dV_t.shape)
            print( "input2dY_t ",input2dY_t.shape)

        #first v1
        #  change to sparse
        input3d_t=self._densetosparse3d(input3d_t)
        xv1=input3d_t
        if self._show_sizes:
            print( "3d sparse size: ",xv1.features.shape)
            print( "3d sparse spatial: ",xv1.spatial_size)

        xv1=self.VCNN1(xv1)
        if self._show_sizes:
            print( "VCNN1 size: ",xv1.features.shape)
            print( "VCNN1 spatial: ",xv1.spatial_size)

        xv1=self._sparsetodense3dVCNN1(xv1)
        if self._show_sizes:
            print( "VCNN1 Dense size: ",xv1.shape)

        xv1_c=self.VFC1(xv1)
        xv1_q=self.VFC1_QUAL(xv1)
        if self._show_sizes:
            print( "VCNN1 Output size: ",xv1_c.shape,xv1_q.shape)
        if self._timing:
            time2 = time.perf_counter()

        # # next v2
        xv2 = input3d_t
        if self._show_sizes:
            print( "3d sparse size: ",xv2.features.shape)
            print( "3d sparse spatial: ",xv2.spatial_size)
        xv2_1 = self.v2_conv1_1(xv2)
        xv2 = input3d_t
        xv2_3 = self.v2_conv3_1(xv2)
        xv2 = input3d_t
        xv2_5 = self.v2_conv5_1(xv2)
        xv2 = self.v2_join([xv2_1,xv2_3,xv2_5])
        xv2 = self.v2_Relu(xv2)
        xv2 = self.v2_drop_1(xv2)


        xv2_1 = self.v2_conv1_2(xv2)
        xv2_3 = self.v2_conv3_2(xv2)


        xv2 = self.v2_join([xv2_1,xv2_3])
        xv2 = self.v2_Relu(xv2)
        xv2 = self.v2_drop_2(xv2)

        xv2 = self.v2_conv3_2(xv2)
        xv2 = self.v2_Relu(xv2)
        xv2 = self.v2_finalmaxpool(xv2)
        xv2 = self.v2_drop_3(xv2)

        xv2=self._sparsetodense3dVCNN2(xv2)
        if self._show_sizes:
            print( "VCNN2 Dense size: ",xv1.shape)
        xv2_c=self.VFC2(xv2)
        xv2_q=self.VFC2_QUAL(xv2)
        if self._show_sizes:
            print( "VCNN2 Output size: ",xv2_c.shape,xv2_q.shape)

        if self._timing:
            time3 = time.perf_counter()

        #first make planes sparse
        xmvU =self._densetosparse2d(input2dU_t)
        xmvV =self._densetosparse2d(input2dV_t)
        xmvY =self._densetosparse2d(input2dY_t)
        if self._show_sizes:
            print( "2d U sparse size: ",xmvU.features.shape)
            print( "2d U sparse spatial: ",xmvU.spatial_size)
            print( "2d V sparse size: ",xmvV.features.shape)
            print( "2d V sparse spatial: ",xmvV.spatial_size)
            print( "2d Y sparse size: ",xmvY.features.shape)
            print( "2d Y sparse spatial: ",xmvY.spatial_size)

        xmvUa = self.alexnet(xmvU)
        xmvVa = self.alexnet(xmvV)
        xmvYa = self.alexnet(xmvY)

        xmvUd=self._sparsetodense2d(xmvUa)
        xmvVd=self._sparsetodense2d(xmvVa)
        xmvYd=self._sparsetodense2d(xmvYa)
        if self._show_sizes:
            print("MV-CNN - U dense: ", xmvUd.shape)
            print("MV-CNN - V dense: ", xmvVd.shape)
            print("MV-CNN - Y dense: ", xmvYd.shape)

        xmv = torch.cat((xmvUd,xmvVd,xmvYd),1)
        if self._show_sizes:
            print("MV-CNN join size: ", xmv.shape)

        xmv_c = self.MVCNNFC(xmv)
        xmv_q = self.MVCNNFC_QUAL(xmv)
        if self._show_sizes:
            print("MV-CNN: ", xmv_c.shape, xmv_q.shape)

        if self._timing:
            time4 = time.perf_counter()

        x = torch.cat((xv1_c,xv2_c,xmv_c),0)
        x = self._softmax(x)
        x_q = torch.cat((xv1_q,xv2_q,xmv_q),0)
        x_q = self._softmax(x_q)
        if self._show_sizes:
            print("Final Sizes: ", x.shape,x_q.shape)


        if self._timing:
            timeend = time.perf_counter()
            print("NETWORK TIMING")
            print("...vcnn1: ",time2-time1)
            print("...vcnn2: ",time3-time2)
            print("...mvcnn: ",time4-time3)
            print("...total time: ",timeend-time1)

        return x,x_q
