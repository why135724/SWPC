import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from models.EEGNet import EEGNet,EEGNet_feature,EEGNet_class,EEGNet_feature_wo_BN
from models.DeepConvNet import DeepConvNet
from models.ShallowConvNet import ShallowConvNet
from EEG_cross_subject_loader_MI import EEG_loader  
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting  
# from models.attention import SingleStudent_multi
#from EEG_cross_subject_loader_augmented import EEG_loader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'


import random
import sys
import time


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# torch.use_deterministic_algorithms(True) 

seed_torch()

def Temporal_Rearrange(x,dataset):
   x_disorder = x.copy()
   if dataset == 'BNCI2014001_2_T_new' or dataset == 'BNCI2014001_2_E_new' or dataset == 'BNCI2014001_T_new':
       x_disorder = np.concatenate((x_disorder[:,:,500:750],x_disorder[:,:,0:250],x_disorder[:,:,250:500]), axis=2)
   if dataset == 'BNCI2014004_T_new':
        x_disorder = np.concatenate((x_disorder[:,:,500:750],x_disorder[:,:,0:250],x_disorder[:,:,250:500]), axis=2)
       # x_disorder = x_disorder[:, [2,1,0], :]
   elif dataset == 'MI2'or dataset == 'MI2_2':
       x_disorder = np.concatenate((x_disorder[:, :, 170:256], x_disorder[:, :, 0:85], x_disorder[:, :, 85:170]),axis=2)
   x = np.concatenate((x,x_disorder),axis = 0)
   y = np.concatenate((np.zeros([len(x_disorder),1]),np.ones([len(x_disorder),1])),axis = 0)
   return x,y



def main(
        test_subj=None,
        learning_rate=None,
        num_iterations=None,
        cuda=None,
        seed=None,
        BN = None,
        RESTING = None,
        test=None,
        test_path=None,
        dataset=None,
        model_name=None,
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
        print('using cuda...')

    if RESTING:
        data = EEG_loader_resting(test_subj=test_subj, dataset=dataset)
    else:
        data = EEG_loader(test_subj=test_subj, dataset=dataset)

    train_x = data.train_x
    train_y = data.train_y

    print('train_x', train_x.shape)  #(1152,22,750)
    print('train_y', train_y.shape)

    tensor_train_x, tensor_train_y = torch.from_numpy(train_x).unsqueeze_(3).permute(0, 3, 2, 1).to(
        torch.float32), torch.from_numpy(train_y).to(torch.long)

    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=64)

    test_x, test_y = data.test_x, data.test_y

    print('test_x', test_x.shape)
    print('test_y', test_y.shape)
    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).unsqueeze_(3).permute(0, 3, 2, 1).to(
        torch.float32), torch.from_numpy(test_y).to(torch.long)

    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset)

    pretext_x = test_x
    pretext_x = pretext_x[:, :, :]
    pretext_x,pretext_y = Temporal_Rearrange(pretext_x,dataset)
    print('pretext_x',pretext_x.shape)
    print('pretext_y', pretext_y.shape)

    tensor_pretext_x, tensor_pretext_y = torch.from_numpy(pretext_x).unsqueeze_(3).permute(0, 3, 2, 1).to(
        torch.float32), torch.squeeze(torch.from_numpy(pretext_y), 1).to(torch.long)

    pretext_dataset = TensorDataset(tensor_pretext_x, tensor_pretext_y)
    pretext_loader = DataLoader(pretext_dataset, batch_size=64)

    # del data, train_x_arr, train_y_arr, train_x_arr_tmp, train_y_arr_tmp, train_x, train_y

    # print(train_x.shape, test_x.shape)
    if model_name == 'DeepConvNet':
        if dataset == 'MI2':
            model = DeepConvNet(4, 22, 256)
        if dataset == 'ERN':
            model = DeepConvNet(2, 56, 260)
        if dataset == 'BNCI201402':
            model = DeepConvNet(2, 15, 1537)
        #if dataset == 'ERP-009-2014':
        #    model = DeepConvNet(2, 16, 32)
    elif model_name == 'ShallowConvNet':
        if dataset == 'MI2':
            model = ShallowConvNet(4, 22, 9152)
        if dataset == 'ERN':
            model = ShallowConvNet(2, 56, 24024)
        if dataset == 'BNCI201402':
            model = ShallowConvNet(2, 15, 41925)
        #if dataset == 'ERP-009-2014':
        #    model = ShallowConvNet(2, 16, 560)
    elif model_name == 'EEGNet':
        if dataset == 'BNCI2014001_2':
            model = EEGNet(22, 376, 2)   #496   872  376  752
            model_feature = EEGNet_feature(22,376)
            model_class = EEGNet_class(376, 2)
            # model = EEGNet_class(376, 2)
            # model = SingleStudent_multi(750)
        elif dataset == 'BNCI2014001':
            model = EEGNet(22, 376, 2)   #496   872  376  752
            model_feature = EEGNet_feature(22,376)
            model_class = EEGNet_class(376, 4)
            model_pretext = EEGNet_class(376, 4)
        elif dataset == 'BNCI2014002':  #496   872  376  752
            model_feature = EEGNet_feature(15,1280)
            model_class = EEGNet_class(1280, 2)
            model_pretext = EEGNet_class(1280, 2)
        elif dataset == 'BNCI2014004':  #496   872  376  752
            model_feature = EEGNet_feature(3,560)
            model_class = EEGNet_class(560, 2)
            model_pretext = EEGNet_class(560, 2)


    model.to(device)
    model_feature.to(device)
    model_class.to(device)

    opt_feature = torch.optim.Adam(model_feature.parameters(),lr = learning_rate)
    opt_class = torch.optim.Adam(model_class.parameters(), lr=learning_rate)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


    model_feature.train()
    model_class.train()
    model.train()
    if not test:
        # Train the model
        for epoch in range(num_iterations):
            print('epoch:', epoch + 1)
            total_loss = 0
            cnt = 0
            for i, (x, y) in enumerate(train_loader):
                if model_name != 'EEGNet':
                     x = x.permute(0, 1, 3, 2)
                # print('x', x.shape) #(64,1,750,22)
                # Forward pass
                x = x.to(device)
                y = y.to(device)
                x_middle = model_feature(x, simsiam=False)
                outputs = model_class(x_middle)
                # outputs = model(x)
                # print('y',y.shape)
                # print('output',outputs.shape)
                loss = criterion(outputs, y)
                total_loss += loss
                cnt += 1

                # Backward and optimize
                opt_feature.zero_grad()
                opt_class.zero_grad()
                opt.zero_grad()
                loss.backward()
                opt_feature.step()
                opt_class.step()
                opt.step()
            out_loss = total_loss / cnt

            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

            if (epoch + 1) % 50 == 0 and epoch != 0:
                # Save the model checkpoint
                if BN:
                    torch.save(model_feature.state_dict(),
                               './runs/' + str(dataset) + 'pre/' + str(model_name) + str(dataset) + 'seed' + str(
                                   seed) + '_test_subj_' + str(test_subj) + '_epoch' + str(epoch + 1) + 'feature_rest_'+str(RESTING)+'.pth')
                else:
                    torch.save(model_feature.state_dict(),
                               './runs/' + str(dataset) + 'pre/' + str(model_name) + str(dataset) + 'seed' + str(
                                   seed) + '_test_subj_' + str(test_subj) + '_epoch' + str(epoch + 1) + 'feature_wo_BN_rest_'+str(RESTING)+'.pth')
                torch.save(model_class.state_dict(),
                           './runs/' + str(dataset) + 'pre/' + str(model_name) + str(dataset) + 'seed' + str(
                               seed) + '_test_subj_' + str(test_subj) + '_epoch' + str(epoch + 1) + 'class_rest_'+str(RESTING)+'.pth')

    else:
        pass

    model_feature.eval()
    model_class.eval()
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        num_0 = 0
        num_0_0 = 0
        num_1 = 0
        num_1_1 = 0
        for x, y in test_loader:
            if model_name != 'EEGNet':
                x = x.permute(0, 1, 3, 2)
            # print('x',x.shape)
            x = x.to(device)
            y = y.to(device)
            # print(y)
            x_middle = model_feature(x, simsiam=False)
            outputs = model_class(x_middle)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        out = 100 * correct / total
        return out


if __name__ == '__main__':
    model_name = 'EEGNet'  # 
    dataset = 'BNCI2014001_2'  #'BNCI2014001_2_T_new'# TODO:  MI2/MI2_2/ERN/BNCI201402/ERP-009-2014
    avg_arr = []
    seed_torch()
    print('dataset',dataset)
    if dataset == 'BNCI2014001_2':
        a = 1
        b = 10
    elif dataset == 'BNCI2014001':
        a = 1
        b = 10
    elif dataset == 'BNCI2014002':
        a = 1
        b = 15
    elif dataset == 'BNCI2014004':
        a = 1
        b = 10
    for i in range(a,b):  
        out_arr = []
        for s in range(0, 1):
            print('subj',i, 'seed', s)
            out = main(
                test_subj=i,
                learning_rate=0.001,
                num_iterations=10,
                cuda=True,
                seed=s,
                BN = True, 
                RESTING=False,
                test=False,
                test_path='./runs/BNCI201402pre/EEGNetBNCI201402seed' + str(s) + '_pretrain_nothrow_model_test_subj_' + str(
                    i) + '_epoch100.ckpt',
                dataset=dataset,
                model_name=model_name,
            )
            out_arr.append(round(out, 3))
        print(out_arr)
        avg_arr.append(np.average(out_arr))
    print(dataset, model_name)
    print(avg_arr)
    print(round(np.average(avg_arr), 3))
    print('pre-train end')



