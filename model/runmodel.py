#!/bin/env python
## IMPORT
# python,numpy
import platform
print(platform.python_version())
import os,sys
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv
import random

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


from networkmodel import Classifier3d


# ===================================================
# TOP-LEVEL PARAMETERS
RUNPROFILER=False
IMAGE_WIDTH=30
IMAGE_HEIGHT=30
IMAGE_DEPTH=30
# ===================================================


def main():

    DEVICE = torch.device("cpu")
    # make an input for testing
    # 3d voxel:
    input3d_t =np.zeros((1,1,30,30,30))
    for i in range(30):
        for j in range(30):
            for k in range(30):
                if random.random() >.8:
                    input3d_t[0][0][i][j][k]=1.0
    # 2d views:
    input2dU_t = np.random.randn(1,1,30,30)
    input2dV_t = np.random.randn(1,1,30,30)
    input2dY_t = np.random.randn(1,1,30,30)


    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 30
    noutput_features = 30
    nplanes = 5
    reps = 1
    # self, inputshape, reps, nin_features, nout_features, nplanes,show_sizes
    model = Classifier3d( (IMAGE_HEIGHT,IMAGE_WIDTH, IMAGE_DEPTH),(IMAGE_HEIGHT,IMAGE_WIDTH), reps,
                           ninput_features, noutput_features,
                           nplanes, True).to(DEVICE)

    # uncomment to dump model
    print ("Loaded model: ",model)

    model.train()
    input3d_t = torch.FloatTensor(input3d_t)
    input2dU_t = torch.FloatTensor(input2dU_t)
    input2dV_t = torch.FloatTensor(input2dV_t)
    input2dY_t = torch.FloatTensor(input2dY_t)

    predict_t = model(input3d_t, input2dU_t, input2dV_t, input2dY_t, 1)

    print(predict_t)


    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:
        print("PROFILER")
        if RUNPROFILER:
            print (prof)


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
