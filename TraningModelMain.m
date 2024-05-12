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

%1D input: Radar Phase
radar_ecg_net_1

%3D input: Heartbeat sigal, output singal in level 3 and level 4 using Daubechies Wavelet
radar_ecg_net_3

%4D input: Radar Phase, Heartbeat sigal, output singal in level 3 and level 4 using Daubechies Wavelet
radar_ecg_net_4