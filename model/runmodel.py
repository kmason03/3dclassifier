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
# this isn't the cleanest, but should work for now
sys.path.insert(1, '../data')
from dataloader import get_net_inputs_mc


# ===================================================
# TOP-LEVEL PARAMETERS
RUNPROFILER=False
IMAGE_WIDTH_3D=1024
IMAGE_HEIGHT_3D=1024
IMAGE_DEPTH_3D=1024
IMAGE_WIDTH_2D=512
IMAGE_HEIGHT_2D=512

DEVICE_IDS=[0,1]
GPUID=DEVICE_IDS[1]
GPUMODE=False
# ===================================================


def main():
    time1 = time.perf_counter()

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(GPUID))
    else:
        DEVICE = torch.device("cpu")
    # test input Loading
    testdata3D, testdata2D, truthlist, truthreco=get_net_inputs_mc(0,-1)
    print(truthlist)
    print(truthreco)

    input3d_t = testdata3D[0]
    input2dU_t = testdata2D[0][0]
    input2dV_t = testdata2D[0][1]
    input2dY_t = testdata2D[0][2]
    truth_t = [truthlist[0],truthreco[0]]
    print(input3d_t.shape)
    print(input2dU_t.shape)

    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 30
    noutput_features = 30
    nplanes = 5
    reps = 1
    time2 = time.perf_counter()
    batchsize = 1
    # self, inputshape, reps, nin_features, nout_features, nplanes,show_sizes, timing
    model = Classifier3d( (IMAGE_HEIGHT_3D,IMAGE_WIDTH_3D, IMAGE_DEPTH_3D),(IMAGE_HEIGHT_2D,IMAGE_WIDTH_2D), reps,
                           ninput_features, noutput_features,
                           nplanes, batchsize, False, True).to(DEVICE)

    # uncomment to dump model
    # print ("Loaded model: ",model)
    time3 = time.perf_counter()

    model.train()
    input3d_t = torch.FloatTensor(input3d_t[None,None,:,:,:]).to(DEVICE)
    input2dU_t = torch.FloatTensor(input2dU_t[None,None,:,:]).to(DEVICE)
    input2dV_t = torch.FloatTensor(input2dV_t[None,None,:,:]).to(DEVICE)
    input2dY_t = torch.FloatTensor(input2dY_t[None,None,:,:]).to(DEVICE)

    predict_t, qual_t = model(input3d_t, input2dU_t, input2dV_t, input2dY_t, 1)
    time4 = time.perf_counter()
    print()
    print("TOTAL TIMING")
    print("...loading data: ", time2-time1)
    print("...loading model: ", time3-time2)
    print("...forward pass: ", time4-time3)
    print("...Total: ", time4-time1)

    # print(predict_t)
    # print(qual_t)
    #
    #
    # with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:
    #     print("PROFILER")
    #     if RUNPROFILER:
    #         print (prof)


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
