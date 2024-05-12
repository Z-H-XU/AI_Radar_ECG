% --------------------------------------------------------------------------------------------------
%
%           Demo software for AI Radar ECG
%                 
%
%            Release ver. 1.0  (May 10, 2024)
%
% --------------------------------------------------------------------------------------------
%
% authors:              Y.X. Wang, X.Y. Wang, and Z.H. Xu.
%
% web page:           https://github.com/Z-H-XU/AI_Radar_ECG
%
% contact:               xuzhihuo@gmail.com
%
% --------------------------------------------------------------------------------------------
% Copyright (c) 2024 NTU.
% Nantong University, China.
% All rights reserved.
% This work should be used for nonprofit purposes only.
% --------------------------------------------------------------------------------------------
%If you utilize this code, please kindly cite the following paper:
%Y.X. Wang, X.Y. Wang, and Z.H. Xu: 'Non-Contact Electrocardiogram via Millimeter-Wave Sensing and Deep Learning'."
%Thank you!
%Wishing you happiness every day, and may the world progress in peace.


clc;clear;close all;

fpath=cd;
addpath(fpath,"API");


load('TraninedModels\ModelsTraningPerformance.mat')

factor=50;

lw=2;
plot(downsample(net1_info.TrainingRMSE,factor),'linewidth',lw)
hold on;
plot(downsample(net3_info.TrainingRMSE,factor),'-.','linewidth',lw)
hold on;
plot(downsample(net4_info.TrainingRMSE,factor),'b:','linewidth',lw)
xlabel("Traning Epochs")
ylabel("Training RMSE")
xlim([0,90])
legend("radar-ecg-net1","radar-ecg-net3","radar-ecg-net4")


figure
plot(downsample(net1_info.TrainingLoss,factor),'linewidth',lw)
hold on;
plot(downsample(net3_info.TrainingLoss,factor),'-.','linewidth',lw)
hold on;
plot(downsample(net4_info.TrainingLoss,factor),'b:','linewidth',lw)
xlim([0,90])
xlabel("Traning Epochs")
ylabel("Training Loss")
legend("radar-ecg-net1","radar-ecg-net3","radar-ecg-net4")