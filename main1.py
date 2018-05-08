# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:35:56 2018

@author: Wei Yuxuan
@Course Project of LOAD PREDICT: Short-Term Load Forecasting (STLF)

main function
using pretrained ANN96_1 (./weights/0.h5~95.h5) for prediction

IO:
    <input>     PSLF_DATA_IN_1.xls
    <output>    2015011942_魏宇轩_PSLF_DATA_OUT_1(2).xls
"""

import numpy as np
import xlrd
from xlutils.copy import copy
import matplotlib.pyplot as plt
from classtrain1 import ANN96_1
from load_data_pre import load_data_to_npy

def writeExcel(vector): 
# write a predicted vector to certain position in an existed EXCEL
    filename = '2015011942_魏宇轩_STLF_DATA_OUT_1.xls'
    rb = xlrd.open_workbook\
    (filename, formatting_info=True)  
    # formatting_info=True, save the original format and data
    wb = copy(rb)  
    ws = wb.get_sheet(0)  
    for i in range(96):
        ws.write(1, 5 + i, vector[i])  
    wb.save(filename) 

def smooth(vector):     # vector smooth
    vlen = len(vector)
    ans = np.zeros((vlen,))
    ans[0] = vector[0]
    ans[vlen-1] = vector[vlen-1]
    for i in range(1,vlen-1):
        ans[i] = 0.2*vector[i-1] + 0.6*vector[i] + 0.2*vector[i+1]
    return ans
 
if __name__ == '__main__':
    npre = load_data_to_npy()   # reload data, find index of date to be predicted
    myANN = ANN96_1()           #  ANN model
    neural = myANN.predict(npre)# load trained weight and predict
    neural_smooth = smooth(neural)  # smooth prediction
    plt.plot(range(96), neural, 'b',range(96), neural_smooth, 'orange')
    writeExcel(neural_smooth)       # write to excel
