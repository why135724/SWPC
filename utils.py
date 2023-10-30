"""
spectral data augmentation
- chaoqi Oct. 29
"""

import torch
import numpy as np
from scipy.signal import spectrogram
import pickle
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
import numpy as np
import scipy.io as sio
from scipy.linalg import fractional_matrix_power
import random




def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = - bound

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    out_ts[out_ts > bound] = bound
    out_ts[out_ts < -bound] = - bound
        
    return out_ts


def EA(x):
	cov = np.zeros((x.shape[0],2, 2))
	for i in range(x.shape[0]):
		cov[i] = np.cov(x[i])
	refEA = np.mean(cov, 0)
	sqrtRefEA=fractional_matrix_power(refEA, -0.5)+ (0.00000001)*np.eye(2)
	XEA = np.zeros(x.shape)
	for i in range(x.shape[0]):
		XEA[i] = np.dot(sqrtRefEA, x[i])
	return  XEA


def add_noise(x, ratio, n_channels,bound):
    """
    Add noise to multiple ts
    Input:
        x: (n_channel, n_length)
    Output:
        x: (n_channel, n_length)
    """
    for i in range(n_channels):  
        if np.random.rand() > ratio:
            mode = np.random.choice(['high','no'])
            x[i, :] = noise_channel(x[i, :], mode=mode, degree=0.05, bound=bound)
    return x


def remove_noise(x,ratio,n_channels,bandpass1,bandpass2,signal_freq,bound):  
    """
    Remove noise from multiple ts
    Input:
        x: (n_channel, n_length)
    Output:
        x: (n_channel, n_length)
    """
    for i in range(n_channels):
        rand = np.random.rand()
        if rand > 0.75:
            x[i, :] = denoise_channel(x[i, :], bandpass1, signal_freq, bound=bound) + \
                      denoise_channel(x[i, :], bandpass2, signal_freq, bound=bound)
        elif rand > 0.5:
            x[i, :] = denoise_channel(x[i, :], bandpass1, signal_freq, bound=bound)
        elif rand > 0.25:
            x[i, :] = denoise_channel(x[i, :], bandpass2, signal_freq, bound=bound)
        else:
            pass
    return x


def crop(x,n_length):
    l = np.random.randint(1, n_length - 1)
    x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

    return x


def augment(x,dataset): 
    t = np.random.rand()
    if t > 0.66:
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001_2_E_new':
            x = add_noise(x, ratio=0.5,n_channels = 22,bound = 0.00025)
        elif dataset == 'MI2_2' or dataset == 'MI2':
            x = add_noise(x, ratio=0.5, n_channels=22, bound=0.00025)
        elif dataset == 'BNCI201402':
            x = add_noise(x, ratio=0.5, n_channels=15, bound=0.00025)
        elif dataset == 'ERN':
            x = add_noise(x, ratio=0.5, n_channels=56, bound=0.00025)
        elif dataset == 'BNCI2015001_T_new':
            x = add_noise(x, ratio=0.5, n_channels=13, bound=0.00025)
    elif t > 0.33:
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001_2_E_new':
            x = remove_noise(x, ratio=0.5, n_channels = 22, bandpass1 = (8, 13), bandpass2 = (20, 30), signal_freq = 125, bound = 0.00025)
        elif dataset == 'MI2_2' or dataset == 'MI2':
            x = remove_noise(x, ratio=0.5, n_channels=22, bandpass1=(8, 13), bandpass2=(20, 30), signal_freq=128,bound=0.00025)
        elif dataset == 'BNCI201402':
            x = remove_noise(x, ratio=0.5, n_channels=15, bandpass1=(8, 13), bandpass2=(20, 30), signal_freq=512,
                             bound=0.00025)
        elif dataset == 'ERN':
            x = remove_noise(x, ratio=0.5, n_channels=56, bandpass1=(8, 13), bandpass2=(20, 30), signal_freq=200,
                             bound=0.00025)
        elif dataset == 'BNCI2015001_T_new':
            x = remove_noise(x, ratio=0.5, n_channels=13, bandpass1=(8, 13), bandpass2=(20, 30), signal_freq=512,
                             bound=0.00025)
    elif t > 0:
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001_2_E_new':
            x = crop(x,n_length=125)
        elif dataset == 'MI2_2' or dataset == 'MI2':
            x = crop(x, n_length=256)
        elif dataset == 'BNCI201402':
            x = crop(x, n_length=1537)
        elif dataset == 'ERN':
            x = crop(x, n_length=260)
        elif dataset == 'BNCI2015001_T_new':
            x = crop(x, n_length=2048)
    # else:
    #     x = x[[1, 0], :]  
    return x


def augment_cutout(x):  
    num_trial = x.shape[0]
    num_channel = x.shape[1]
    num_timepoints = x.shape[2]
    for i in range(num_trial):
        num_cutout = np.random.randint(low = 1, high=4)
        for j in range(num_cutout):
            channel_cutoff = np.random.randint(low = 0, high=num_channel)
            x[i,channel_cutoff,:] = np.zeros((1,num_timepoints))
    return x


class EEG_loader_augment_resting(): 

    def __init__(self, test_subj=-1, dataset=None):

        if dataset == 'BNCI2014001_2':
            dataset_test = 'BNCI2014001_2_E'
        test_subj = test_subj
        data_folder = './data/' + str(dataset_test)

        test_x = []
        test_y = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        x = np.array(mat['X'])
        # x = np.moveaxis(np.array(mat['x']), -1, 0)
        if dataset == 'BNCI2014001_2':
            x_motor = x[:, :, 750:1000]
            x_motor = x_motor[:,:,::2]
            x_resting = x[:, :, 0:250]
            x_resting = x_resting[:,:,::2]
            x = np.concatenate([x_motor, x_resting], axis=0)
        # print(x)

        y = np.concatenate([np.zeros((len(x_motor),)),np.ones((len(x_resting),))],axis=0)

        print(y.shape)
        test_x.append(x)
        test_y.append(y)

        if dataset == 'BNCI2014001_2':
            dataset_train = 'BNCI2014001_2_T'
        test_subj = test_subj
        data_folder = './data/' + str(dataset_train)

        train_x = []
        train_y = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        # x = np.moveaxis(np.array(mat['x']), -1, 0)
        x = np.array(mat['X'])
        if dataset == 'BNCI2014001_2':
            x_motor = x[:, :, 750:1000]
            x_motor = x_motor[:,:,::2]
            x_resting = x[:, :, 0:250]
            x_resting = x_resting[:,:,::2]
            x = np.concatenate([x_motor, x_resting], axis=0)

        # y = np.concatenate([np.zeros((2*len(x_motor), )), np.ones((2*len(x_resting), ))], axis=0)
        y = np.concatenate([np.zeros((len(x_motor),)), np.ones((len(x_resting),))], axis=0)
        train_x.append(x)
        train_y.append(y)

        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        idx = list(range(len(train_x)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]
        train_x_aug1 = train_x.copy()
        train_x_aug2 = train_x.copy()
        for v in range(len(train_x)):
            # x_aug1[v,:,:] = augment(x[v,:,:],dataset)
            # x_aug2[v,:,:] = augment(x[v,:,:],dataset)
            train_x_aug1[v,:,:] = train_x[v,:,:]
            train_x_aug2[v,:,:] = train_x[v,:,:]
        
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        self.train_x = train_x
        self.train_x_aug1 = train_x_aug1
        self.train_x_aug2 = train_x_aug2
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y




def sliding_windows(x,y,step,point_start,point_end,upper_limit):
   x_all = []
   y_all = []

   for i in range(int((upper_limit-point_end)/step)):
       x_all.append(x[:, :, point_start+i*step:point_end+i*step])
       y_all.append(i)

   x_new = np.concatenate(x_all, axis=0)
   y_new = np.concatenate(y_all, axis=0)



   return x_new,y_new


class EEG_loader_augment():  

    def __init__(self, test_subj=-1, dataset=None):

        if dataset == 'BNCI2014001_2':
            dataset_test = 'BNCI2014001_2_E'
        elif dataset == 'BNCI2014001':
            dataset_test = 'BNCI2014001_E'
        elif dataset == 'BNCI2014002':
            dataset_test = 'BNCI2014002_E'
        elif dataset == 'BNCI2014004':
            dataset_test = 'BNCI2014004_E'

        test_subj = test_subj
        data_folder = './data/' + str(dataset_test)
        data_folder = '/mnt/ssd2/hywu/BCICIV_2a_gdf/' + str(dataset_test)
        test_x = []
        test_y = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        x = np.array(mat['X'])
        # x = np.moveaxis(np.array(mat['x']), -1, 0)
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001':
            x = x[:, :, 0:750]
        elif dataset == 'BNCI2014002':
            x = x[:, :, 0:2560]
        # print(x)
        y = np.array(mat['y'])
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001' or dataset == 'BNCI2014004':
            y_num = []
            for i in range(len(y)):
                if y[i][:9] == 'left_hand':
                    y_num.append(0)
                elif y[i][:10] == 'right_hand':
                    y_num.append(1)
                elif y[i][:4] == 'feet':
                    y_num.append(2)
                elif y[i][:6] == 'tongue':
                    y_num.append(3)
            y = np.array(y_num)
        elif dataset == 'BNCI2014002':
            y_num = []
            for i in range(len(y)):
                if y[i][:10] == 'right_hand':
                    y_num.append(0)
                elif y[i][:4] == 'feet':
                    y_num.append(1)
            y = np.array(y_num)
        # print(y.shape)
        test_x.append(x)
        test_y.append(y)

        if dataset == 'BNCI2014001_2':
            dataset_train = 'BNCI2014001_2_T'
        elif dataset == 'BNCI2014001':
            dataset_train = 'BNCI2014001_T'
        elif dataset == 'BNCI2014002':
            dataset_train = 'BNCI2014002_T'
        elif dataset == 'BNCI2014004':
            dataset_train = 'BNCI2014004_T'

        test_subj = test_subj
        data_folder = './data/' + str(dataset_train)
        data_folder = '/mnt/ssd2/hywu/BCICIV_2a_gdf/' + str(dataset_train)
        train_x = []
        train_y = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        # x = np.moveaxis(np.array(mat['x']), -1, 0)
        x = np.array(mat['X'])
        y = np.array(mat['y'])
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001':
            x = x[:, :, 0:750]
            y_num = []
            for i in range(len(y)):
                if y[i][:9] == 'left_hand':
                    y_num.append(0)
                elif y[i][:10] == 'right_hand':
                    y_num.append(1)
                elif y[i][:4] == 'feet':
                    y_num.append(2)
                elif y[i][:6] == 'tongue':
                    y_num.append(3)
            y = np.array(y_num)
        elif dataset == 'BNCI2014004':
            y_num = []
            for i in range(len(y)):
                if y[i][:9] == 'left_hand':
                    y_num.append(0)
                elif y[i][:10] == 'right_hand':
                    y_num.append(1)
                elif y[i][:4] == 'feet':
                    y_num.append(2)
                elif y[i][:6] == 'tongue':
                    y_num.append(3)
            y = np.array(y_num)
        elif dataset == 'BNCI2014002':
            x = x[:, :, 0:2560]
            y_num = []
            for i in range(len(y)):
                if y[i][:10] == 'right_hand':
                    y_num.append(0)
                elif y[i][:4] == 'feet':
                    y_num.append(1)
            y = np.array(y_num)

        train_x.append(x)
        train_y.append(y)

        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        idx = list(range(len(train_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]
        train_x_aug1 = train_x.copy()
        train_x_aug2 = train_x.copy()
        for v in range(len(train_x)):
            train_x_aug1[v,:,:] = augment(train_x_aug1[v,:,:],dataset)
            train_x_aug2[v,:,:] = augment(train_x_aug2[v,:,:],dataset)
            # train_x_aug1[v, :, :] = train_x[v, :, :]
            # train_x_aug2[v, :, :] = train_x[v, :, :]

        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)



        self.train_x = train_x
        self.train_y = train_y
        self.train_x_aug1 = train_x_aug1
        self.train_x_aug2 = train_x_aug2
        self.test_x = test_x
        self.test_y = test_y


class EEG_loader_augment_resting_cross(): 

    def __init__(self, test_subj=-1, dataset=None):

        if dataset == 'BNCI2014001_2':
            dataset_test = 'BNCI2014001_2_E'
        test_subj = test_subj
        data_folder = './data/' + str(dataset_test)

        test_x = []
        test_y = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        x = np.array(mat['X'])
        # x = np.moveaxis(np.array(mat['x']), -1, 0)
        if dataset == 'BNCI2014001_2':
            x_motor = x[:, :, 750:1000]
            x_resting = x[:, :, 0:250]
            x = np.concatenate([x_motor, x_resting], axis=0)
        # print(x)
        y = np.concatenate([np.zeros((len(x_motor),)),np.ones((len(x_resting),))],axis=0)

        # print(y.shape)
        test_x.append(x)
        test_y.append(y)

        if dataset == 'BNCI2014001_2':
            dataset_train = 'BNCI2014001_2_E'
        test_subj = test_subj
        data_folder = './data/' + str(dataset_train)

        train_x_temp = []
        train_y_temp = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        a = 1
        if dataset == 'BNCI2014001_2':
            k = 10

        for i in range(a, k):
            if i == test_subj:
                continue
            mat = sio.loadmat(data_folder + "/" + prefix + str(i) + ".mat")
            x = np.array(mat['X'])

            print(x.shape)
            if dataset == 'BNCI2014001_2':
                x_motor = x[:, :, 750:1000]
                x_resting = x[:, :, 0:250]
                x = np.concatenate([x_motor, x_resting], axis=0)

            y = np.concatenate([np.zeros((len(x_motor),)), np.ones((len(x_resting),))], axis=0)
            print('x.shape',x.shape)
            train_x_temp.append(x)
            train_y_temp.append(y)

        print('len_trian_x',len(train_x_temp))


        for j in range(a, k-2): 
            print('j',j)
            if j == 1:
                train_x = np.concatenate( [train_x_temp[j-1],train_x_temp[j]] , axis=0 )
                train_y = np.concatenate([train_y_temp[j - 1], train_y_temp[j]], axis=0)
            else:
                train_x = np.concatenate( [train_x,train_x_temp[j]] , axis=0 )
                train_y = np.concatenate([train_y, train_y_temp[j]], axis=0)

        print('train_x',train_x.shape)
        idx = list(range(len(train_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        self.train_x = train_x
        self.train_y = train_y
        self.train_x_aug1 = train_x
        self.train_x_aug2 = train_x
        self.test_x = test_x
        self.test_y = test_y


class EEG_loader_augment_cross(): 

    def __init__(self, test_subj=-1, dataset=None):

        if dataset == 'BNCI2014001_2':
            dataset_test = 'BNCI2014001_2_E'
        elif dataset == 'BNCI2014001':
            dataset_test = 'BNCI2014001_E'
        elif dataset == 'BNCI2014002':
            dataset_test = 'BNCI2014002_E'
        elif dataset == 'BNCI2014004':
            dataset_test = 'BNCI2014004_E'

        test_subj = test_subj
        data_folder = '/mnt/ssd2/hywu/BCICIV_2a_gdf/' + str(dataset_test)
        test_x = []
        test_y = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        x = np.array(mat['X'])
        # x = np.moveaxis(np.array(mat['x']), -1, 0)
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001':
            x = x[:, :, 0:750]
        elif dataset == 'BNCI2014002':
            x = x[:, :, 0:2560]

        # print(x)
        y = np.array(mat['y'])
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001' or dataset == 'BNCI2014004':
            y_num = []
            for i in range(len(y)):
                if y[i][:9] == 'left_hand':
                    y_num.append(0)
                elif y[i][:10] == 'right_hand':
                    y_num.append(1)
                elif y[i][:4] == 'feet':
                    y_num.append(2)
                elif y[i][:6] == 'tongue':
                    y_num.append(3)
            y = np.array(y_num)
        elif dataset == 'BNCI2014002':
            y_num = []
            for i in range(len(y)):
                if y[i][:10] == 'right_hand':
                    y_num.append(0)
                elif y[i][:4] == 'feet':
                    y_num.append(1)
            y = np.array(y_num)

        # print(y.shape)
        test_x.append(x)
        test_y.append(y)

        if dataset == 'BNCI2014001_2':
            dataset_train = 'BNCI2014001_2_E'
        elif dataset == 'BNCI2014001':
            dataset_train = 'BNCI2014001_E'
        elif dataset == 'BNCI2014002':
            dataset_train = 'BNCI2014002_E'
        elif dataset == 'BNCI2014004':
            dataset_train = 'BNCI2014004_E'

        test_subj = test_subj
        data_folder = '/mnt/ssd2/hywu/BCICIV_2a_gdf/' + str(dataset_train)
        train_x_temp = []
        train_y_temp = []

        if dataset == 'ERN':
            prefix = 's'
        else:
            prefix = 'A'

        a = 1
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001' or dataset == 'BNCI2014004':
            k = 10
        elif dataset == 'BNCI2014002':
            k = 15

        for i in range(a, k):
            if i == test_subj:
                continue
            mat = sio.loadmat(data_folder + "/" + prefix + str(i) + ".mat")
            x = np.array(mat['X'])
            y = np.array(mat['y'])
            if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001' or dataset == 'BNCI2014004':
                y_num = []
                for i in range(len(y)):
                    if y[i][:9] == 'left_hand':
                        y_num.append(0)
                    elif y[i][:10] == 'right_hand':
                        y_num.append(1)
                    elif y[i][:4] == 'feet':
                        y_num.append(2)
                    elif y[i][:6] == 'tongue':
                        y_num.append(3)
                y = np.array(y_num)
            elif dataset == 'BNCI2014002':
                y_num = []
                for i in range(len(y)):
                    if y[i][:10] == 'right_hand':
                        y_num.append(0)
                    elif y[i][:4] == 'feet':
                        y_num.append(1)
                y = np.array(y_num)
            print(x.shape)

            print('x.shape',x.shape)
            train_x_temp.append(x)
            train_y_temp.append(y)

        print('len_trian_x',len(train_x_temp))

        for j in range(a, k-2):
            print('j',j)
            if j == 1:
                train_x = np.concatenate( [train_x_temp[j-1],train_x_temp[j]] , axis=0 )
                train_y = np.concatenate([train_y_temp[j - 1], train_y_temp[j]], axis=0)
            else:
                train_x = np.concatenate( [train_x,train_x_temp[j]] , axis=0 )
                train_y = np.concatenate([train_y, train_y_temp[j]], axis=0)

        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001':
            train_x = train_x[:, :, 0:750]
        elif dataset == 'BNCI2014002':
            train_x = train_x[:, :, 0:2560]


        print('train_x',train_x.shape)
        idx = list(range(len(train_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        self.train_x = train_x
        self.train_y = train_y
        self.train_x_aug1 = train_x
        self.train_x_aug2 = train_x
        self.test_x = test_x
        self.test_y = test_y


