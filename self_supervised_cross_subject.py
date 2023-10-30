import torch
from utils import EEG_loader_augment_cross,EEG_loader_augment_resting_cross
# 测试自监督的效果
import numpy as np
import torch.nn as nn
import random
import os
import argparse

from models.EEGNet import EEGNet_feature,EEGNet_class
from loss import MoCo, SimCLR, BYOL, OurLoss, SimSiam,SupCon
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# python self_supervised_within_subject.py --model SimSiam --resting
# python self_supervised_within_subject.py --model SimCLR --resting
# python self_supervised_within_subject.py --model SupCon --resting

# python self_supervised_cross_subject.py --model ContraWR --lr 2e-4 --T 2
# python self_supervised_within_subject.py --model BYOL --resting
# python self_supervised_within_subject.py --model MoCo --resting
# cd /mnt/ssd2/hywu/SWRA

# nohup python -u self_supervised_cross_subject.py > BNCI2014004_cross_test.log 2>&1 &
# kill -9 77986


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# https://blog.csdn.net/lgzlgz3102/article/details/122711169


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def random_construct_queue(y,train_x,train_y,device,k_encoder): #构造对应的queue
    k_encoder.eval()
    y = y.cpu().numpy()
    # print(y)
    code = []
    for i in range(len(y)):
        if y[i] == 0:
            code.append(1)
        else:
            code.append(0)
    # print('code',code)
    # print('train_y',train_y)
    train_y = train_y
    indexs_0 = np.where(train_y == 0)[0].tolist()
    # print('indexs_0',indexs_0)
    indexs_1 = np.where(train_y == 1)[0].tolist()
    x_code = np.empty((0,train_x.shape[1], train_x.shape[2]))
    for j in code:
        if j == 0:
            # print(train_x[[random.choice(indexs_0)],:,:].shape)
            x_code = np.concatenate([x_code,train_x[[random.choice(indexs_1)],:,:]],axis = 0)
        elif j == 1:
            # print(train_x[[random.choice(indexs_1)], :, :].shape)
            x_code = np.concatenate([x_code,train_x[[random.choice(indexs_0)],:,:]],axis = 0)
    # print('x_code',x_code.shape)
    x_code = torch.from_numpy(x_code).unsqueeze_(3).permute(0, 3, 2, 1).to(
        torch.float32)
    # print('x_code',x_code.shape)
    x_code = x_code.to(device)
    x_code = k_encoder(x_code, simsiam=False)
    # print('x_code', x_code.shape)
    k_encoder.train()
    return x_code

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
parser.add_argument('--lr', type=float, default=0.5e-3, help="learning rate")
parser.add_argument('--n_dim', type=int, default=376, help="hidden units (for SHHS, 256, for Sleep, 128, for BNCI2014001, 376,for MI2_2,128 for BNCI201402, 768)")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
parser.add_argument('--training', type=int, default=10, help="training subject")
parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient")
parser.add_argument('--model', type=str, default='ContraWR', help="which model")
parser.add_argument('--T', type=float, default=0.3, help="T")
parser.add_argument('--sigma', type=float, default=2.0, help="sigma")
parser.add_argument('--delta', type=float, default=0.2, help="delta")
parser.add_argument('--dataset', type=str, default='BNCI2014001', help="dataset") #MI2
parser.add_argument('--model_name', type=str, default='EEGNet', help="model")
parser.add_argument('--epoch_pretrain', type=int, default=100, help="number of pretrain epochs")
parser.add_argument('--test_subj', type=int, default=5, help="number of sub")
parser.add_argument('--channels', type=int, default=22, help="number of channels")
parser.add_argument('--num_class', type=int, default=4, help="number of chass")
parser.add_argument('--sub_start', type=int, default=1, help="first sub")
parser.add_argument('--sub_end', type=int, default=10, help="last sub")
parser.add_argument('--resting', action='store_true', default=False, help='.')
args = parser.parse_args()

def Pretext(q_encoder, k_encoder, optimizer,optimizer_k, Epoch, criterion, pretext_loader, test_loader,device,args,train_x,train_y):
    q_encoder.train();
    k_encoder.train()
    # print(q_encoder)
    global queue
    global queue_ptr
    global n_queue
    acc_max = 0
    step = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

    all_loss, acc_score = [], []
    for epoch in range(Epoch):
        for index, (aug1, aug2, _, y) in enumerate(pretext_loader):
            aug1, aug2 = aug1.to(device), aug2.to(device)
            if args.model in ['BYOL']:
                emb_aug1 = q_encoder(aug1, simsiam=False)
                emb_aug2 = k_encoder(aug2, simsiam=False)
            elif args.model in ['SimCLR']:
                emb_aug1 = q_encoder(aug1, simsiam=False)
                emb_aug2 = q_encoder(aug2, simsiam=False)
            elif args.model in ['ContraWR']:
                emb_aug1 = q_encoder(aug1,simsiam=False)
                emb_aug2 = k_encoder(aug2, simsiam=False)
                # queue = random_construct_queue(y, train_x, train_y, device, k_encoder)  # 构造专用的queue
            elif args.model in ['MoCo']:
                emb_aug1 = q_encoder(aug1, simsiam=False)
                emb_aug2 = k_encoder(aug2, simsiam=False)
            elif args.model in ['SimSiam']:
                emb_aug1, proj1 = q_encoder(aug1, simsiam=True)
                emb_aug2, proj2 = q_encoder(aug2, simsiam=True)
            elif args.model in ['SupCon']:
                emb_aug1 = q_encoder(aug1, simsiam=False)
                emb_aug2 = k_encoder(aug2, simsiam=False)

            # backpropagation
            if args.model == 'MoCo':
                loss = criterion(emb_aug1, emb_aug2, queue)
                if queue_ptr + emb_aug2.shape[0] > n_queue:
                    queue[queue_ptr:] = emb_aug2[:n_queue-queue_ptr]
                    queue[:queue_ptr+emb_aug2.shape[0]-n_queue] = emb_aug2[-(queue_ptr+emb_aug2.shape[0]-n_queue):]
                    queue_ptr = (queue_ptr + emb_aug2.shape[0]) % n_queue
                else:
                    queue[queue_ptr:queue_ptr+emb_aug2.shape[0]] = emb_aug2
            elif args.model == 'SimSiam':
                loss = criterion(proj1, proj2, emb_aug1, emb_aug2)
            elif args.model == 'SupCon':
                features = torch.cat([emb_aug1.unsqueeze(1), emb_aug1.unsqueeze(1)], dim=1)
                # print('features',features.shape)
                loss = criterion(features,y)
            else:
                loss = criterion(emb_aug1, emb_aug2)


            all_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # only update encoder_q

            # exponential moving average (EMA)
            for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                param_k.data = param_k.data * args.m + param_q.data * (1. - args.m)

            # scheduler.step(sum(all_loss[-50:]))
            # step += 1

        # print the lastest result
        print('epoch: {}'.format(epoch))
        print(evaluate(q_encoder,test_loader,device,args))
        acc = evaluate(q_encoder,test_loader,device,args)
        if acc > acc_max:
            acc_max = acc
    return round(acc_max,2),round(acc,2)



def evaluate(q_encoder, test_loader,device,args):
    # freeze
    q_encoder.eval()

    model_class = EEGNet_class(args.n_dim,args.num_class)
    # #
    # q_encoder = EEGNet_feature(args.channels,args.n_dim)
    # q_encoder.load_state_dict(torch.load('./runs/' + str(args.dataset) + 'pre/' + str(args.model_name) + str(args.dataset) + 'seed' + str(
    #                            seed) + '_test_subj_' + str(args.test_subj) + '_epoch' + str(args.epoch_pretrain) + 'feature.pth'))
    # q_encoder.to(device)
    # q_encoder.eval()

    # 加载预训练模型的参数
    # model_class.load_state_dict(torch.load('./runs/' + str(args.dataset) + 'pre/' + str(args.model_name) + str(args.dataset) + 'seed' + str(
    #                            seed) + '_test_subj_' + str(args.test_subj) + '_epoch' + str(args.epoch_pretrain) + 'class.pth'))
    model_class.load_state_dict(
        torch.load('./runs/' + str(args.dataset) + 'pre/' + str(args.model_name) + str(args.dataset) + 'seed' + str(
            seed) + '_test_subj_' + str(args.test_subj) + '_epoch' + str(args.epoch_pretrain) + 'class_rest_' + str(
            args.resting) + '.pth'))
    model_class.to(device)
    model_class.eval()

    # torch.save(q_encoder.state_dict(),
    #            './runs/' + str(args.dataset) + 'pre/' + str(args.model_name) + str(args.dataset) + 'seed' + str(
    #                seed) + '_test_subj_' + str(args.test_subj) + '_epoch' + str(100) + 'feature_rest_' + str(
    #                args.resting) + '.pth')

    # for param_q, param_k in model_class.named_parameters():
    #     print(param_k)

    with torch.no_grad():
        correct = 0
        total = 0
        num_0 = 0
        num_0_0 = 0
        num_1 = 0
        num_1_1 = 0
        for x, y in test_loader:
            if args.model_name != 'EEGNet':
                x = x.permute(0, 1, 3, 2)
            x = x.to(device)
            y = y.to(device)
            # print(y)
            x_middle = q_encoder(x,simsiam=False)
            outputs = model_class(x_middle)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if args.dataset == 'ERN' or args.dataset == 'ERP-009-2014':
                for b in range(len(y)):
                    if y[b] == 0:
                        num_0 += 1
                        if predicted[b] == 0:
                            num_0_0 += 1
                    elif y[b] == 1:
                        num_1 += 1
                        if predicted[b] == 1:
                            num_1_1 += 1

        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        if args.dataset == 'ERN' or args.dataset == 'ERP-009-2014':
            out_ERP = 100 * (0.5 * (num_0_0 / num_0) + 0.5 * (num_1_1 / num_1))
        out = 100 * correct / total
        if args.dataset == 'ERN' or args.dataset == 'ERP-009-2014':
            out = out_ERP
    # print (cm, 'accuracy', res)
    # print (cm)
    # q_encoder.train()
    return out

# nohup python -u self_supervised_cross_subject.py > BNCI2014002_cross_test.log 2>&1 &

if __name__ == '__main__':

    # model_list = ["MoCo", "SimCLR", "BYOL", "ContraWR", "SimSiam", "SupCon"]
    model_list = ["ContraWR"]
    for model in model_list:
        args.dataset = 'BNCI2014001_2'
        args.model = model
        if args.model == "ContraWR":
            args.lr = 2e-4
            args.T = 2
        if args.dataset == 'BNCI2014001_2':
            args.channels = 22
            args.num_class = 2
            args.sub_start = 1
            args.sub_end = 10
            args.n_dim = 376
        elif args.dataset == 'BNCI2014001':
            args.channels = 22
            args.num_class = 4
            args.sub_start = 1
            args.sub_end = 10
            args.n_dim = 376
        elif args.dataset == 'BNCI2014002':
            args.channels = 15
            args.num_class = 2
            args.sub_start = 1
            args.sub_end = 15
            args.n_dim = 1280
        elif args.dataset == 'BNCI2014004':
            args.channels = 3
            args.num_class = 2
            args.sub_start = 1
            args.sub_end = 10
            args.n_dim = 560
        acc_total = []
        acc_low_total = []
        for i in range(args.sub_start, args.sub_end):  #MI2_2 1,10
            print('i',i)
            args.test_subj = i
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('device:', device)

            # set random seed
            seed = 0
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # torch.backends.cudnn.benchmark = True

            global queue
            global queue_ptr
            global n_queue

            # dataset = 'BNCI2014001_2_T_new'
            # model_name = 'EEGNet'
            # epoch_pretrain = 100
            # test_subj = 9
            print('args.dataset',args.dataset)
            print('args.test_subj', args.test_subj)
            print('args.resting',args.resting)
            print('model', model)
            q_encoder = EEGNet_feature(args.channels,args.n_dim)

            q_encoder.load_state_dict(torch.load('./runs/' + str(args.dataset) + 'pre/' + str(args.model_name) + str(args.dataset) + 'seed' + str(
                           seed) + '_test_subj_' + str(args.test_subj) + '_epoch' + str(args.epoch_pretrain) + 'feature_rest_' + str(
                           args.resting) + '.pth'))

            # for name, parameter in q_encoder.named_parameters():
            #     parameter.requires_grad = False

            # model_feature.to(device)
            # model_feature.eval()
            # q_encoder = EEGNet_feature(22,args.n_dim)

            q_encoder.to(device)

            k_encoder = EEGNet_feature(args.channels,args.n_dim)
            # k_encoder.load_state_dict(torch.load('./runs/' + str(args.dataset) + 'pre/' + str(args.model_name) + str(args.dataset) + 'seed' + str(
            #                            seed) + '_test_subj_' + str(args.test_subj) + '_epoch' + str(args.epoch_pretrain) + 'feature.pth'))
            k_encoder.to(device)

            for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False  # not update by gradient
                # param_q.requires_grad = False

            optimizer = torch.optim.Adam(q_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_k = torch.optim.Adam(k_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # assign contrastive loss function
            if args.model == 'ContraWR':
                criterion = OurLoss(device, args.delta, args.sigma, args.T).to(device)
            elif args.model == 'MoCo':
                criterion = MoCo(device).to(device)
                queue_ptr, n_queue = 0, 4096
                queue = torch.tensor(np.random.rand(n_queue, args.n_dim), dtype=torch.float).to(device)
            elif args.model == 'SimCLR':
                criterion = SimCLR(device).to(device)
            elif args.model == 'BYOL':
                criterion = BYOL(device).to(device)
            elif args.model == 'SimSiam':
                criterion = SimSiam(device).to(device)
            elif args.model == 'SupCon':
                criterion = SupCon(device).to(device)

            if args.resting:
                data = EEG_loader_augment_resting_cross(test_subj=args.test_subj, dataset=args.dataset)
            else:
                data = EEG_loader_augment_cross(test_subj=args.test_subj, dataset=args.dataset)

            train_x_aug1 = data.train_x_aug1
            train_x_aug2 = data.train_x_aug2
            train_x = data.train_x
            train_y = data.train_y
            test_x = data.test_x
            test_y = data.test_y

            print('train_x_aug1.shape',train_x_aug1.shape)  #(144,22,750)
            print('train_x_aug2.shape',train_x_aug2.shape)
            print('train_x.shape',train_x.shape)  #(144,22,750)
            print('test_x.shape',test_x.shape)
            print('test_y_.shape',test_y.shape)  #(144,22,750)

            tensor_train_x_aug1, tensor_train_x_aug2,tensor_train_x,tensor_train_y = torch.from_numpy(train_x_aug1).unsqueeze_(3).permute(0, 3, 2, 1).to(
                torch.float32), torch.from_numpy(train_x_aug2).unsqueeze_(3).permute(0, 3, 2, 1).to(
                torch.float32),torch.from_numpy(train_x).unsqueeze_(3).permute(0, 3, 2, 1).to(
                torch.float32),torch.from_numpy(train_y).to(torch.long)

            train_dataset = TensorDataset(tensor_train_x_aug1, tensor_train_x_aug2,tensor_train_x,tensor_train_y)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

            test_x, test_y = data.test_x, data.test_y

            print('test_x', test_x.shape)
            print('test_y', test_y.shape)

            tensor_test_x, tensor_test_y = torch.from_numpy(test_x).unsqueeze_(3).permute(0, 3, 2, 1).to(
                torch.float32), torch.from_numpy(test_y).to(torch.long)

            test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
            test_loader = DataLoader(test_dataset)

            tensor_test_x, tensor_test_y = torch.from_numpy(test_x).unsqueeze_(3).permute(0, 3, 2, 1).to(
                torch.float32), torch.from_numpy(test_y).to(torch.long)

            pretext_dataset = TensorDataset(tensor_test_x,tensor_test_x,tensor_test_x, tensor_test_y)
            pretext_loader = DataLoader(pretext_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

       #     # optimize
            acc,acc_low = Pretext(q_encoder, k_encoder, optimizer, optimizer_k, args.epochs, criterion, pretext_loader, test_loader,device,args,data.train_x, data.train_y)
            acc_total.append(acc)
            acc_low_total.append(acc_low)

        print('all_sub_acc', acc_total)
        print('mean_sub_acc', round(np.mean(acc_total), 2))
        print('all_sub_acc_low', acc_low_total)
        print('mean_sub_acc_low', round(np.mean(acc_low_total), 2))