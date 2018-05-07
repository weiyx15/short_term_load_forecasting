# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:23:45 2018

@author: Wei Yuxuan
@Course Project of LOAD PREDICT: Short-Term Load Forecasting (STLF)

prepocess load data
save load data to file "load.npy"
drop out bad data and save index to "bad_index.npy"
"""

import numpy as np
import xlrd

def load_data_to_npy():
    workbook = xlrd.open_workbook('STLF_DATA_IN_1.xls') # raw data
    load_sheet = workbook.sheet_by_index(0)             # sheet0: load data
    nrows = load_sheet.nrows - 1                    # rows of load data
    ncols = load_sheet.ncols - 1                    # cols of load data
    load = np.zeros((nrows, ncols), dtype=np.float) # load matrix without bad data
    bad_index = list()                              # rows of bad data
    Is_bad = 0                                      # if there is '0' in rows
    for i in range(nrows):
        try:
            Is_bad = 0
            for j in range(1,ncols+1):
                if load_sheet.cell_value(i,j) == 0: # there is a '0'
                    bad_index.append(i)             # it is also a line of bad data
                    Is_bad = 1
                    break
            if Is_bad == 0:
                load[i,:] = load_sheet.row_values(i)[1:ncols+1]
        except ValueError:                          # null cell
            bad_index.append(i)
    
    np.save("load.npy", load)
    np.save("bad_index.npy", bad_index)
    return nrows                              # return numbers of data entries