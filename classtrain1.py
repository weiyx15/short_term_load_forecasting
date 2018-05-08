# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:33:01 2018
@author: Wei Yuxuan
@Course Project of LOAD PREDICT: Short-Term Load Forecasting (STLF)
based on classtrain.py
Revision: remove "today_t-1" data source from training and prediction
DATA
last_week_t, last_week_t-1, yesterday_t, yesterday_t-1
and is_weekend_one_hot (1:weekday, 0:weekend) for yesterday, today
"""
import numpy as np
from keras import Sequential
from keras.layers import Dense 
from keras import optimizers
import time
import os

class ANN96_1:
    # global constant
    xcols = 6# last_week_t, last_week_t-1, yesterday_t, yesterday_t-1
    norm1 = [0,1,2,3]                     # list of load normalization
    
    def __init__(self):     # construction function
        # build model
        self.model = Sequential()
        self.model.add(Dense(8, input_shape=(ANN96_1.xcols,), activation='relu'))
        self.model.add(Dense(1, activation=None))
        self.model.summary()
        # loss = mean squared error
        sgd = optimizers.SGD(lr=0.05)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
        # load data
        self.load = np.load('load.npy')
        # and is_weekend_one_hot (1:weekday, 0:weekend) for yesterday, today
        self.loadmax = self.load.max()                  # max load ever
        load0 = self.load[self.load>0]                  # delete the 0s
        self.loadmin = load0.min()                      # min load ever (not 0)
        self.nepoch = 100                   # default training epochs
    
#    def ANNmodel(self):     # ANN model structure for all 96 classifiers
#         # 2 layers dense network
#        self.model = Sequential()
#        self.model.add(Dense(8, input_shape=(ANN96.xcols,), activation='relu'))
#        self.model.add(Dense(1, activation=None))
#        self.model.summary()
#        # loss = mean squared error
#        sgd = optimizers.SGD(lr=0.5)
#        self.model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
    
    def setEpochs(self, Nepoch):            # set training epochs
        self.nepoch = Nepoch
    
#    def load_data(self):
#        self.load = np.load('load.npy')
#        # and is_weekend_one_hot (1:weekday, 0:weekend) for yesterday, today
#        self.loadmax = self.load.max()                  # max load ever
#        load0 = self.load[self.load>0]                  # delete the 0s
#        self.loadmin = load0.min()                      # min load ever (not 0)
    
    def train(self, n):             # use 0~n-1 date for training
        start_time = time.time()
        for j in range(96):
            x = np.zeros((1, ANN96_1.xcols), dtype=np.float)          # train data
            y = list()                                              # train label
            index = list()# original date index, for Mon/Tues recognition in sample_weight
            if j == 0:
                # prepare data
                for i in range(8, n-1): 
                    xt = np.zeros((1, ANN96_1.xcols), dtype=np.float)     
                    if self.load[i-7, j]!=0 and self.load[i-8, 95]!=0 and \
                    self.load[i-1, j]!=0 and self.load[i-2, 95]!=0 \
                    and self.load[i, j]!=0:
                        xt[0,0] = self.load[i-7, j]
                        xt[0,1] = self.load[i-8, 95]
                        xt[0,2] = self.load[i-1, j]
                        xt[0,3] = self.load[i-2, 95]
                        if (i-1) % 7 == 3 or (i-1) % 7 == 4:
                            xt[0,4] = 0
                        else:
                            xt[0,4] = 1
                        if i % 7 == 3 or i % 5 == 4:
                            xt[0,5] = 0
                        else:
                            xt[0,5] = 1
                        if x[0, 0] == 0:
                            x = xt
                        else:
                            x = np.concatenate((x, xt), axis=0)
                        y.append(self.load[i, j])
                        index.append(i)
            else:
                for i in range(7, n-1):
                    xt = np.zeros((1, ANN96_1.xcols), dtype=np.float)       # temporary variable
                    if self.load[i-7, j]!=0 and self.load[i-7, j-1]!=0 \
                    and self.load[i-1, j]!=0 and self.load[i-1, j-1]!=0 \
                    and self.load[i, j]!=0:
                        xt[0,0] = self.load[i-7, j]
                        xt[0,1] = self.load[i-7, j-1]
                        xt[0,2] = self.load[i-1, j]
                        xt[0,3] = self.load[i-1, j-1]
                        if (i-1) % 7 == 3 or (i-1) % 7 == 4:
                            xt[0,4] = 0
                        else:
                            xt[0,4] = 1
                        if i % 7 == 3 or i % 5 == 4:
                            xt[0,5] = 0
                        else:
                            xt[0,5] = 1
                        if x[0, 0] == 0:
                            x = xt
                        else:
                            x = np.concatenate((x, xt), axis=0)
                        y.append(self.load[i, j])
                        index.append(i)
            # training
            # normalization
            nsample = x.shape[0]            # number of training samples
            for i in range(nsample):
                for k in ANN96_1.norm1:
                    x[i, k] = (x[i, k] - self.loadmin)\
                    / (self.loadmax - self.loadmin)
            y = (y - self.loadmin) / (self.loadmax - self.loadmin)
            # sample weight: 
            # 1. the closer the date, the larger the weight in loss function
            # 2. the weight of Monday and Tuesday (days to be predicted) sample is amplified
            sw = np.zeros((nsample,))
            for i in range(nsample):
                sw[i] = 0.998**(nsample-i)
                if index[i] % 7 == 5 or index[i] % 7 == 6:      # Mon or Tues
                    sw[i] = sw[i] * 2
                elif index[i] % 7 == 3 or index[i] % 7 == 4:    # delete weekend
                    sw[i] = 0
            # train
            if j == 0:          # j=0, nepoch *= 10
                history = self.model.fit\
            (x, y, batch_size=32, epochs=10*self.nepoch, \
             sample_weight=sw, verbose=0)
            else:
                history = self.model.fit\
            (x, y, batch_size=32, epochs=self.nepoch,sample_weight=sw, verbose=1)
            # verbose=0, not to show training details
            # save model
            self.model.save_weights(os.path.join('new_weights',str(j))+'.h5')
            # evaluate model by mse
            mse = history.history['mean_squared_error']
            last_mse = mse[len(mse)-1]
            print(str(j) + '/96: ' + str(last_mse))
        end_time = time.time()
        use_time = np.ceil((end_time - start_time) / 60)
        print(str(use_time) + ' min used in trainng process')
    
    def predict(self, n):       # predict the nth date
        neural = np.zeros((96,))
        xp = np.zeros((1, ANN96_1.xcols), dtype=np.float)         # predict data
        prediction = 0
        for j in range(96):
            if j == 0:
                xp[0,0] = self.load[n-7, j]
                xp[0,1] = self.load[n-8, 95]
                xp[0,2] = self.load[n-1, j]
                xp[0,3] = self.load[n-2, 95]
                if (n-1) % 7 == 3 or (n-1) % 7 == 4:
                    xp[0,4] = 0
                else:
                    xp[0,4] = 1
                if n % 7 == 3 or n % 7 == 4:
                    xp[0,5] = 0
                else:
                    xp[0,5] = 1
                for k in ANN96_1.norm1:
                    xp[0, k] = (xp[0, k] - self.loadmin)\
                    / (self.loadmax - self.loadmin)
            else:
                xp[0,0] = self.load[n-7, j]
                xp[0,1] = self.load[n-7, j-1]
                xp[0,2] = self.load[n-1, j]
                xp[0,3] = self.load[n-1, j-1]
                if (n-1) % 7 == 3 or (n-1) % 7 == 4:
                    xp[0,4] = 0
                else:
                    xp[0,4] = 1
                if n % 7 == 3 or n % 7 == 4:
                    xp[0,5] = 0
                else:
                    xp[0,5] = 1
                for k in ANN96_1.norm1:
                    xp[0, k] = (xp[0, k] - self.loadmin)\
                    / (self.loadmax - self.loadmin)
            # predict
            self.model.load_weights(os.path.join('new_weights',str(j)+'.h5'))
            prediction = self.model.predict(xp)
            prediction = \
            (self.loadmax - self.loadmin) * prediction + self.loadmin
            neural[j] = prediction
        return neural