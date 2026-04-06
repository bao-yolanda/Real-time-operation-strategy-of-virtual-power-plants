%% Standardize parameters for PV, ES, EV only (NO IPP and TCL)
% This version only generates constraints for PV, ES, EV resources

%% NOFDER计算（只有PV、ES、EV）
NOFDER = 1 + 1 + NOFEV;  % PV=1, ES=1, EV=NOFEV

%% 标准化参数矩阵初始化
param_std.energy_init = zeros(NOFDER, 1);
param_std.energy_upper_limit = zeros(NOFDER, NOFSLOTS);
param_std.energy_lower_limit = zeros(NOFDER, NOFSLOTS);
param_std.energy_end = zeros(NOFDER, 1);
param_std.power_dis_upper_limit = zeros(NOFDER, NOFSLOTS);
param_std.power_dis_lower_limit = zeros(NOFDER, NOFSLOTS);
param_std.power_ch_upper_limit = zeros(NOFDER, NOFSLOTS);
param_std.power_ch_lower_limit = zeros(NOFDER, NOFSLOTS);
param_std.theta = ones(NOFDER, 1);
param_std.eta_ch = zeros(NOFDER, NOFSLOTS);
param_std.eta_dis = zeros(NOFDER, NOFSLOTS);
param_std.pr_dis = zeros(NOFDER, 1);
param_std.pr_ch = zeros(NOFDER, 1);
param_std.wOmiga = zeros(NOFDER, NOFSLOTS);

%% PV (第1个资源)
param_std.energy_init(1) = 0;
param_std.energy_upper_limit(1, :) = param.power_dis_upper_limit_pv';
param_std.energy_lower_limit(1, :) = 0;
param_std.energy_end(1) = 0;
param_std.power_dis_upper_limit(1, :) = param.power_dis_upper_limit_pv';
param_std.power_dis_lower_limit(1, :) = 0;
param_std.power_ch_upper_limit(1, :) = 0;
param_std.power_ch_lower_limit(1, :) = 0;
param_std.theta(1) = 1;
param_std.eta_ch(1, :) = 1;
param_std.eta_dis(1, :) = 1;
param_std.pr_dis(1) = 0;
param_std.pr_ch(1) = 0;

%% ES (第2个资源)
param_std.energy_init(2) = param.energy_init_es;
param_std.energy_upper_limit(2, :) = param.energy_upper_limit_es;
param_std.energy_lower_limit(2, :) = param.energy_lower_limit_es;
param_std.energy_end(2) = param.energy_init_es;  % 最终能量等于初始能量
param_std.power_dis_upper_limit(2, :) = param.power_dis_upper_limit_es;
param_std.power_dis_lower_limit(2, :) = param.power_dis_lower_limit_es;
param_std.power_ch_upper_limit(2, :) = param.power_ch_upper_limit_es;
param_std.power_ch_lower_limit(2, :) = param.power_ch_lower_limit_es;
param_std.theta(2) = param.theta_es;
param_std.eta_ch(2, :) = param.eta_ch_es;
param_std.eta_dis(2, :) = param.eta_dis_es;
param_std.pr_dis(2) = param.pr_dis_es;
param_std.pr_ch(2) = param.pr_ch_es;

%% EV (第3~(NOFEV+2)个资源)
for ev_idx = 1:NOFEV
    der_idx = 2 + ev_idx;
    param_std.energy_init(der_idx) = param.energy_init_ev;
    param_std.energy_upper_limit(der_idx, :) = param.energy_upper_limit_ev;
    param_std.energy_lower_limit(der_idx, :) = param.energy_lower_limit_ev;
    param_std.energy_end(der_idx) = param.energy_end_ev;
    param_std.power_dis_upper_limit(der_idx, :) = param.power_dis_upper_limit_ev;
    param_std.power_dis_lower_limit(der_idx, :) = param.power_dis_lower_limit_ev;
    param_std.power_ch_upper_limit(der_idx, :) = param.power_ch_upper_limit_ev;
    param_std.power_ch_lower_limit(der_idx, :) = param.power_ch_lower_limit_ev;
    param_std.theta(der_idx) = param.theta_ev;
    param_std.eta_ch(der_idx, :) = param.eta_ch_ev;
    param_std.eta_dis(der_idx, :) = param.eta_dis_ev;
    param_std.pr_dis(der_idx) = param.pr_dis_ev;
    param_std.pr_ch(der_idx) = param.pr_ch_ev;
end

%% 非调频资源索引（只有PV、ES、EV，全部参与调频，所以为空）
% 明确设置为空数组（double类型）
param.index_none_reg = double.empty(0, 1);

%% 将NOFDER保存到param中，方便后续检查
param.NOFDER = NOFDER;

%% 清理临时变量
clear ev_idx der_idx
