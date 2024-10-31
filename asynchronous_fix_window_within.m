clear all
close all
clc
t = 0;
tic;

ACC_TOTAL = [];
FPR_TOTAL = [];
Label_total = [];
Predict_total = [];
window_length_all = [250,250,250,250,250,250,250,250,250]; 
for v = 1:1 %控制随机种子数量
    v
    ACC = [];
    FINAL = [];
    ITR_ALL = [];
    FPR = [];
    dataFolder_1 = ['./'];
    files_1 = dir([dataFolder_1 'train_A*.mat']);
    dataFolder_2 = ['./'];
    files_2 = dir([dataFolder_2 'test_A*.mat']);

    for s = 1:5 %length(files) 
        label_all = [];
        s
        tic
        load([dataFolder_1 files_1(s).name]);
        X = permute(X,[2,3,1]);
        y_new = [];
        for c = 1:length(y)
            if y(c,1) == 'l'
               y_new = [y_new;1];
            end
            if y(c,1) == 'r'
               y_new = [y_new;2];
            end
            if y(c,1) == 'f'
               y_new = [y_new;3];
            end
            if y(c,1) == 't'
               y_new = [y_new;4];
            end
        end
        y = y_new;
        X_train_rest = X(:,251:500,1:72); %分割静息态
        X_train_image = X(:,751:1000,:); %分割运动想象态
        
        X_train = cat(3,X_train_rest,X_train_image); %baseline 
     
       y_train = cat(1,ones(72,1)+2,y);  %baseline

        clear X y;

        load([dataFolder_2 files_2(s).name]); 
        X = permute(X,[2,3,1]);
        y_new = [];
        for c = 1:length(y)
            if y(c,1) == 'l'
               y_new = [y_new;1];
            end
            if y(c,1) == 'r'
               y_new = [y_new;2];
            end
        end
        y = y_new;

        X_test = X(:,1:1750,:);
        X_test_total = [];
        for n = 1:144  %180
            X_test_total = cat(2,X_test_total,X_test(:,:,n));
        end

        y_test = y;        
    %     y_test = cat(1,y_test,ones(36,1)+2); 

        CSPm=4;     %定义CSP-m参数
        sampleRate = 128;
        startTime=3;
        data_length = 250; 
        num_class = 2; %类别数
        rho = 1.1;
        window_length = window_length_all(v); %窗口长度
        %% 训练模型
        [fTrain,W] = CSPfeature_train(X_train(:,:,:),y_train);
        model=libsvmtrain(y_train,fTrain,'-b 1 -c 2 -g 0.1250'); %训练模型  
        
        [num_point_test,num_channel_test,num_trials_test] = size(X_test);

        for i = 0:10:144*1750-window_length %180*1500-250 = 215750  先得到滑动窗在线的所有输出，然后根据这个输出找何时开始接收数据并分类
%             i
           [fTest] = CSPfeature_test(X_test_total(:,i+1:i+window_length,:),W);
           [predictlabel_s,ac_s,decv]=libsvmpredict(y_test(1),fTest,model,'-b 1');
           label_all = [label_all;predictlabel_s i+window_length];
        end

       label_all = label_all;
        pred_fpr = [];
        pred = [];
        for k = 1:144
            s = [];
            t = [];
            for m = 1:length(label_all)
                if 750+1750*(k-1)<label_all(m,2)&&label_all(m,2)<1750*k
                    s = [s label_all(m,1)];
                end
                if 1750*(k-1)<label_all(m,2)&&label_all(m,2)<500+1750*(k-1)
                    t = [t;label_all(m,1)];
                end
            end
            if isempty(s)
                pred = [pred;3];
            else
%                  s
                 pred = [pred;s(end)];
                 %pred = [pred;mode(s)];
            end
            if isempty(t)
                pred_fpr = [pred_fpr;3];
            else
%                 pred = [pred;s(1)];
                  pred_fpr = [pred_fpr;1];
            end
        end
        
        num_true = 0;
        num_image = 0;
        for b = 1:length(y_test)
             if y_test(b) ~= 3
                num_image = (num_image+1);
                if(pred(b) == y_test(b))
                    num_true = (num_true+1);
                end
            end
        end
        acc = (num_true/num_image);
        ACC = [ACC;100*acc];
         
        num_true = 0;
        num_image = 0;
        for b = 1:length(y_test)
             if y_test(b) ~= 3
                num_image = (num_image+1);
                if(pred_fpr(b) ~= 3)
                    num_true = (num_true+1);
                end
            end
        end
        fpr = (num_true/num_image);
       Label_total = [Label_total y_test];
       Predict_total = [Predict_total pred];
    end
    ACC_TOTAL = [ACC_TOTAL;mean(ACC)];
    FPR_TOTAL = [FPR_TOTAL;mean(FPR)];
end
ACC = [ACC;mean(ACC)]; %最终精度加上平均值
