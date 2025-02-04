import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from models.EEGNet import EEGNet_feature,EEGNet_class,TransformerModel
from models.DeepConvNet import DeepConvNet
from models.ShallowConvNet import ShallowConvNet
#预训练EEGNET的代码
# self_supervised_MI_DAN_DeepDA
from EEG_cross_subject_loader_MI import EEG_loader  #改这里
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting  #改这里
# from models.attention import SingleStudent_multi
#from EEG_cross_subject_loader_augmented import EEG_loader

# cd /mnt/ssd2/hywu/SWRA
# nohup python -u pre_train_cross_subject.py > BNCI2014001_2_cross_pre.log 2>&1 &
# kill -9 77986
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# cd /mnt/data2/hywu/SWRA
# python pre_train_cross_subject.py
# conda activate ws

import random
import sys
import time


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

def split_dataset(x):

    #根据数据集划分出生成的数据集

    x_middle = []
    y_middle = []

    for i in range(10):
        x_middle.append(x[:,0+i:240+i,:])
        y_middle.append(x[:,[240+i],:])

    x_new = np.concatenate(x_middle,axis=0)
    y_new = np.concatenate(y_middle,axis=0)

    return x_new,y_new

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

    #
    print('train') #只用audio
    train_x = data.train_x
    print('x',train_x.shape)
    train_x = np.transpose(train_x, (0,2,1))
    print('train_x',train_x.shape)  #5524,25,768

    traineeg, labeleeg = split_dataset(train_x)
    print('traineeg', traineeg.shape)  #5524, 150, 33
    print('labeleeg', labeleeg.shape)  # 5524, 150, 33

    tensor_testeeg, tensor_testlabel = torch.from_numpy(traineeg).to(torch.float32), torch.from_numpy(labeleeg).to(
        torch.float32),

    train_dataset = TensorDataset(tensor_testeeg,tensor_testlabel)
    gpt_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    gpt_test_loader = DataLoader(train_dataset, batch_size=64)

    seed = 0

    device = torch.device('cuda')
    model = TransformerModel()
    model.to(device)

    learning_rate = 0.0005
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.MSELoss()

    for epoch in range(10):
        print('epoch:', epoch + 1)
        total_loss = 0
        cnt = 0
        for i, (x, x1) in enumerate(gpt_train_loader):
            # Forward pass
            x = x.to(device)
            x1 = x1.to(device)
            # print('y',y)
            outputs = model(x)
            loss = criterion(outputs, x1)
            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, 40, out_loss))
        # # 找qk矩阵的排序
        model.eval()
        print('test')

        with torch.no_grad():
            total_loss = 0
            cnt = 0
            qk_final = 0
            gpt_outputs = []
            for (x, x1) in gpt_test_loader:
                x = x.to(device)
                # print('x',x.shape)
                x1 = x1.to(device)
                # print('y1',y)
                outputs = model(x)
                gpt_outputs.append(outputs.detach().cpu().numpy())
                # print('qk',qk.sum())
                loss = criterion(outputs, x1)
                total_loss += loss
                cnt += 1
            # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
            out = total_loss / cnt
            # print(total)
            # print(qk_final)
            print('MSE: {:.4f}'.format(out))

    gpt_outputs_final = np.concatenate(gpt_outputs,axis=0)

    print('gpt',gpt_outputs_final.shape)  #10000,1,33

    gpt_outputs_final = gpt_outputs_final.reshape(144, 10,gpt_outputs_final.shape[2])

    print('gpt', gpt_outputs_final.shape)  # 10000,1,33

    train_x[:,240:,:] = gpt_outputs_final
    train_x_gpt = np.transpose(train_x, (0,2,1))
    print('gpt',train_x_gpt.shape)

    del train_x

    print('train')
    #训练集
    train_x = data.train_x
    train_y = data.train_y

    # 加不加数据增强
    train_x = np.concatenate([train_x,train_x_gpt],axis=0)
    train_y = np.concatenate([train_y,train_y],axis=0)

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
             # 250:120 750:376
            model_feature = EEGNet_feature(22,120)
            model_class = EEGNet_class(120, 2)
        elif dataset == 'BNCI2014001':
            model_feature = EEGNet_feature(22,120)
            model_class = EEGNet_class(120, 4)
        elif dataset == 'BNCI2014002':
            model_feature = EEGNet_feature(15,256)
            model_class = EEGNet_class(256, 2)
        elif dataset == 'BNCI2014004':
            model_feature = EEGNet_feature(3,560)
            model_class = EEGNet_class(560, 2)


    model_feature.to(device)
    model_class.to(device)

    opt_feature = torch.optim.Adam(model_feature.parameters(),lr = learning_rate)
    opt_class = torch.optim.Adam(model_class.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


    model_feature.train()
    model_class.train()
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
                x_middle = model_feature(x)
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
                loss.backward()
                opt_feature.step()
                opt_class.step()
            out_loss = total_loss / cnt

            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for x, y in test_loader:
            #         if model_name != 'EEGNet':
            #             x = x.permute(0, 1, 3, 2)
            #         # print('x',x.shape)
            #         x = x.to(device)
            #         y = y.to(device)
            #         # print(y)
            #         x_middle = model_feature(x)
            #         outputs = model_class(x_middle)
            #         _, predicted = torch.max(outputs.data, 1)
            #         # total += y.size(0)
            #         # correct += (predicted == y).sum().item()
            #         total += (y == 0).sum().item()
            #         correct += (predicted == y & y == 0).sum().item()
            #     # model
            #     # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
            #     out = 100 * correct / total
            #     print(out)

    else:
        pass

    model_feature.eval()
    model_class.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            if model_name != 'EEGNet':
                x = x.permute(0, 1, 3, 2)
            # print('x',x.shape)
            x = x.to(device)
            y = y.to(device)
            # print(y)
            x_middle = model_feature(x)
            outputs = model_class(x_middle)
            _, predicted = torch.max(outputs.data, 1)
            # total += y.size(0)
            # correct += (predicted == y).sum().item()
            total += (y == 0).sum().item()
            correct += (predicted == y & y == 0).sum().item()
        # model
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        out = 100 * correct / total
        return out


if __name__ == '__main__':
    model_name = 'EEGNet'  # TODO: 改一下
    dataset = 'BNCI2014001_2'  #'BNCI2014001_2_T_new'# TODO: 改一下   MI2/MI2_2/ERN/BNCI201402/ERP-009-2014
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
    for i in range(a,b):   #subj in range (1,10) MI2 共9个被试/subj in range (0,16) ERN 共16个被试/subj in range (1,15) BNCI201402 共14个被试/subj in range (1,11) ERP-009-2014 共10个被试
        out_arr = []
        for s in range(0, 1):
            print('subj',i, 'seed', s)
            out = main(
                test_subj=i,
                learning_rate=0.001,
                num_iterations=100,
                cuda=True,
                seed=s,
                BN = True, #是否打开BN层
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

    # [80.556, 66.667, 80.556, 48.611, 81.944, 55.556, 51.389, 88.889, 66.667]
    # 68.982

    # [68.056, 54.167, 72.222, 9.722, 83.333, 70.833, 80.556, 90.278, 33.333]
    # 62.5

    # [75.0, 70.833, 93.056, 34.722, 87.5, 69.444, 76.389, 66.667, 61.111]
    # 70.525

    # [87.5, 29.167, 77.78, 50, 93.56, 47.22, 65.27, 72.22,37.5] GPT
    # 62.5

    # [ 87.5, 27.778, 77.78, 58.33, 91.667, 59.772, 65.27, 80.556,

# [56.944, 76.389, 87.5, 93.056, 65.278, 76.389, 86.111, 90.278, 95.833]
# 80.864

# [43.056, 66.667, 79.167, 93.056, 56.944, 77.778, 73.611, 90.278, 90.278]
# 74.537

# [36.111, 70.833, 77.778, 58.333, 84.722, 70.833, 77.778, 86.111, 86.111]
# 72.068


# [43.056, 66.667, 73.611, 51.389, 72.222, 70.833, 50.0, 97.222, 87.5]
# 68.056





