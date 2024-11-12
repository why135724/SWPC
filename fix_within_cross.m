clear all;
close all;
clc;


% 初始化变量
ACC_TOTAL = [];
FPR_TOTAL = [];
Label_total = [];
Predict_total = [];
window_length_all = 250;

EA = false;  %控制是否打开EA

% 控制随机种子数量（这里只循环一次，所以v的值固定为1）
for v = 1 
    ACC = [];
    FPR = [];
    
    % 定义数据文件夹和文件列表
    dataFolder_train = './';
    files_train = dir([dataFolder_train 'train_A*.mat']);
    XRaw=[]; yAll=[]; XAlignE=[];
    
    for s=1:5 %length(files)
        s
        load([dataFolder_train files_train(s).name]);
        X = permute(X,[2,3,1]); 
        y = convertLabels(y);
        
        XRaw=cat(3,XRaw,X); %XRaw(22,1751,1296)  未EA的数据
        yAll=cat(1,yAll,y); %1296*1
        nTrials=length(y);

        RtE=mean(covariances(X),3); % reference state, Euclidean space
        sqrtRtE=RtE^(-1/2); 
        XE=nan(size(X,1),size(X,2),nTrials);
        for j=1:nTrials
            XE(:,:,j)=sqrtRtE*X(:,:,j); %改这个
        end
        XAlignE=cat(3,XAlignE,XE); %XAlignE(22,1751,1296)  EA后的数据
    end    

    % 遍历每个文件进行处理
    for s = 1:5
        nTrials = 144;
        yt=yAll((s-1)*nTrials+1:s*nTrials);
        ys=yAll([1:(s-1)*nTrials s*nTrials+1:end]); %1152 1
        XtRaw=XRaw(:,:,(s-1)*nTrials+1:s*nTrials);
        XsRaw=XRaw(:,:,[1:(s-1)*nTrials s*nTrials+1:end]);  %22 1751 1152
        XtAlignE=XAlignE(:,:,(s-1)*nTrials+1:s*nTrials);  %22 1751 144
        XsAlignE=XAlignE(:,:,[1:(s-1)*nTrials s*nTrials+1:end]);  %22 1751 1152
        
        % 加载训练数据并转换标签
%         data_train = load([dataFolder_train files_train(s).name]);
        if EA
            X_train = XsAlignE;
        else
            X_train = XsRaw;
        end
        y_train = ys;
        
        [rows, cols,high] = size(X_train);
        % 提取训练特征
        X_train_rest = X_train(:, 251:500, 1:high/2);
        X_train_image = X_train(:, 751:1000, :);
        X_train = cat(3, X_train_rest, X_train_image);
        y_train = cat(1, ones(high/2, 1) + 2, y_train);
        
        % 加载测试数据并转换标签
        if EA
            X_test = XtAlignE;
        else
            X_test = XtRaw;
        end
        y_test = yt;
        
        % 提取测试特征并展平
        X_test_total = reshape(X_test, size(X_test, 1), []);
        
        % 训练模型
        CSPm = 4;
        [fTrain, W] = CSPfeature_train(X_train, y_train);
        model = libsvmtrain(y_train, fTrain, '-b 1 -c 2 -g 0.1250');
        
        % 滑动窗口进行预测
        window_length = window_length_all(v);
        num_windows = size(X_test_total, 2) - window_length + 1;
        label_all = zeros(num_windows, 2);
        
        for i = 1:num_windows
            fTest = CSPfeature_test(X_test_total(:, i:i+window_length-1), W);
            [predictlabel_s, ~, ~] = libsvmpredict(y_test(1), fTest, model, '-b 1');
            label_all(i, :) = [predictlabel_s, i + window_length - 1];
        end
        
        % 处理预测结果
        pred = processPredictions(label_all, size(X_test, 3));
        
        % 计算准确率和假正率
        [acc, fpr] = calculateMetrics(y_test, pred);
        
        % 累积结果
        ACC = [ACC; 100 * acc];
        FPR = [FPR; fpr];
        Label_total = [Label_total; y_test];
        Predict_total = [Predict_total; pred];
    end
    
    % 计算平均准确率和假正率
    ACC_TOTAL = [ACC_TOTAL; mean(ACC)];
    FPR_TOTAL = [FPR_TOTAL; mean(FPR)];
end

% 显示最终平均准确率
disp(['最终平均准确率: ', num2str(mean(ACC_TOTAL)), '%']);

% 辅助函数（与原代码相同）...
% 辅助函数：转换标签
function y_new = convertLabels(y)
    y_new = zeros(length(y), 1);
    for i = 1:length(y)
        switch y(i, 1)
            case 'l'
                y_new(i) = 1;
            case 'r'
                y_new(i) = 2;
            case 'f'
                y_new(i) = 3;
            case 't'
                y_new(i) = 4;
        end
    end
end

% 辅助函数：处理预测结果
function pred = processPredictions(label_all, num_trials)
    pred = zeros(num_trials, 1);
    for k = 1:num_trials
        trial_labels = label_all(label_all(:, 2) >= 750 + 1750 * (k - 1) & ...
                                 label_all(:, 2) < 1750 * k, 1);
        if isempty(trial_labels)
            pred(k) = 3; % 假设3表示未知或静息态
        else
            pred(k) = trial_labels(end); % 使用最后一个窗口的预测结果
        end
    end
end

% 辅助函数：计算准确率和假正率
function [acc, fpr] = calculateMetrics(y_test, pred)
    num_true = 0;
    num_image = 0;
    num_fpr_true = 0;
    for i = 1:length(y_test)
        if y_test(i) ~= 3
            num_image = num_image + 1;
            if pred(i) == y_test(i)
                num_true = num_true + 1;
            end
            if pred(i) ~= 3
                num_fpr_true = num_fpr_true + 1;
            end
        end
    end
    acc = num_true / num_image;
    fpr = num_fpr_true / num_image;
end