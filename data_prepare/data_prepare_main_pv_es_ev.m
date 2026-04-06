%% Prepare the parameters for the problem (PV + ES + EV only)
clc;
clear;

day_price = 22;
M = 1e6;
delta_t_req = 0.5;

%% Read the RegD signal data
NOFSLOTS = 24;
day_reg = day_price + 1;
hour_init = 0;

% 调频信号离散细粒度
granularity = 0.1;
% 调频信号处理
data_prepare_regd;

%% Parameters for each resource (只包含PV、ES、EV)
data_prepare_parameters_pv_es_ev;

%% Standardize the parameters
data_prepare_std_pv_es_ev;

% Resource names (只有PV、ES、EV)
param.resource_names = ["pv", "es", "ev"];

% Resource numbers (只有PV、ES、EV)
param.resource_range = [[1, 1]; [2, 2]; [3, NOFEV + 2]];

%% Market prices and other parameters
delta_t = 1;

% Read the regulation market price data
filename = 'regulation_market_results.xlsx';
sheet = 'regulation_market_results';
start_row = (day_price-1) * 24 + hour_init + 2;
xlRange = "G" + start_row + ":H" + (start_row + NOFSLOTS - 1);
param.price_reg = xlsread(filename, sheet, xlRange);

NOFSCEN = length(param.hourly_Distribution(1, :));
param.s_perf = 0.984;

% Read the system energy price data
filename = 'rt_hrl_lmps.xlsx';
sheet = 'rt_hrl_lmps';
xlRange = "I" + start_row + ":I" + (start_row + NOFSLOTS - 1);
param.price_e = xlsread(filename, sheet, xlRange);

clear price filename sheet xlRange start_row signal_length idx jdx

% 保存新的.mat文件（只有PV、ES、EV）
save("param_day_" + day_price + "_pv_es_ev.mat");
disp(['数据已保存到: param_day_' + day_price + '_pv_es_ev.mat']);
