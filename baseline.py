# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:19:16 2018

@author: Wei Yuxuan
@Course Project of LOAD PREDICT: Short-Term Load Forecasting (STLF)

baseline method: ratio smoothing
"""

import numpy as np

def curve_smooth(n, alpha):
    load = np.load("load.npy")
    ncol = load.shape[1]
    trainset = np.zeros((14,ncol))      # two weeks normalized training data
    trainset[0,:] = load[n-7,:]/max(load[n-7,:])    # the last related day
    trainset[1,:] = load[n-14,:]/max(load[n-14,:])  # the second last related day
    unrelated = list();                             # index of unrelated day
    for i in range(1,7):
        unrelated.append(i)
    for i in range(8,14):
        unrelated.append(i)
    for i in range(12):
        trainset[i+2,:] = load[n-unrelated[i],:]/max(load[n-unrelated[i],:])
    normal = np.zeros((ncol,))
    for i in range(14):
        normal = normal + alpha*(1-alpha)**i*trainset[i,:]  # normalized curve
    A0 = 0                              # peak smooth factor (first period)
    A1 = 0                              # peak smooth factor (second period)
    A0list = [1,2,3,4,5,6]
    A1list = [8,9,10,11,12,13]
    for i in range(6):
        A0 = A0 + alpha*(1-alpha)**i*max(load[n-A0list[i],:])
        A1 = A1 + alpha*(1-alpha)**i*max(load[n-A1list[i],:])
    return max(load[n-7,:])/A1*A0*normal