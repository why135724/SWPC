import numpy as np
import scipy.io as sio


class EEG_loader():

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
            x = x[:,:,0:2560]
        # print(x)
        y = np.array(mat['y'])
        if dataset == 'BNCI2014001_2' or dataset == 'BNCI2014001'or dataset == 'BNCI2014004':
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
            if dataset == 'BNCI2014001_2' or  dataset == 'BNCI2014001':
                x = x[:, :, 0:750]
            elif dataset == 'BNCI2014001_2':
                x = x[:,:,0:2560]

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
        self.test_x = test_x
        self.test_y = test_y