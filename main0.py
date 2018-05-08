# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:35:56 2018

@author: Wei Yuxuan
@Course Project of LOAD PREDICT: Short-Term Load Forecasting (STLF)

FOR TESTING
compare baseline (ratio smoothing) & ANN96/ANN96_1 (neural network)
decide to use ANN96_1 ultimately

FOR TRAINING
learning rate: 0.5
training epoch: 200
"""

import numpy as np
import matplotlib.pyplot as plt
from load_data_pre import load_data_to_npy
from baseline import curve_smooth
from classtrain1 import ANN96_1
from main1 import smooth

#def two_method(n):
#    curve = curve_smooth(n, 0.99)
#    err = (curve - load[n,:]) / load[n,:]
#    precision_baseline = (1-np.sqrt(np.dot(err,err)/err.shape[0]))
#    
#    myANN = ANN96()
#    myANN.train(n)
#    neural = myANN.predict(n)
#    err = (neural - load[n,:]) / load[n,:]
#    precision_neural = (1-np.sqrt(np.dot(err,err)/err.shape[0]))
#    
#    
#    plt.plot(range(96), load[n,:], 'b', range(96), curve, 'orange',\
#             range(96), neural, 'g')
#    print('ratio smooth: ' + str(precision_baseline))
#    print('neural network: ' + str(precision_neural))
#
#def neural_method(nt, np):      # 0~nt-1: train data, np: predict date
#    myANN = ANN96()
#    myANN.setEpochs(100)
#    myANN.train(nt)
#    neural = myANN.predict(np)
#    err = (neural - load[np,:]) / load[np,:]
#    precision_neural = (1-np.sqrt(np.dot(err,err)/err.shape[0]))
#    plt.plot(range(96), load[np,:], 'b', range(96), neural, 'g')
#    print('neural network: ' + str(precision_neural))


if __name__ == '__main__':
    npre = load_data_to_npy()   # reload data, find index of date to be predicted
    load = np.load("load.npy")
    n1 = 1944 - 42
    n2 = 1944 - 7
    # train
    myANN = ANN96_1()
    myANN.setEpochs(15)
    myANN.train(1944)                # no longer training
    # predict
#    npre = n2
#    xn = load[npre,:]
#    neural = myANN.predict(npre)
#    neural_smooth = smooth(neural)  # smooth prediction
#    err = (neural_smooth - xn) / xn
#    precision_neural = (1-np.sqrt(np.dot(err,err)/err.shape[0]))
#    plt.plot(range(96),xn,'b',range(96), neural_smooth, 'g')
#    plt.savefig("virtual_predict.png")
#    print('neural network: ' + str(precision_neural))
#    curve = curve_smooth(npre, 0.9)
#    err = (curve - xn) / xn
#    precision_baseline = (1-np.sqrt(np.dot(err,err)/err.shape[0]))
#    print('ratio smooth: ' + str(precision_baseline))
#    plt.plot(range(96),xn,'b',range(96), neural_smooth, 'g',\
#             range(96), curve, 'r')
#    plt.savefig('method_compare.png')