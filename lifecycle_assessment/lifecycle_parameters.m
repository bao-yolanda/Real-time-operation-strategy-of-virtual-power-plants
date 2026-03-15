%% Lifecycle Assessment Parameters
% 全生命周期投资评估参数定义
%
% Defines investment costs, degradation parameters, O&M costs, and
% financial parameters for the full lifecycle assessment of the VPP.
%
% Resource configuration (from data_prepare_parameters.m):
%   PV  : 1 MWp photovoltaic system (index 1)
%   ES  : 2 MWh / ±1 MW Li-ion energy storage (index 2)
%   EV  : 40 vehicles × 60 kWh, ±22 kW V2G (indices 3-42)
%   TCL : 3 temperature-controlled buildings (indices 43-45)
%   IPP : 10 industrial production processes (indices 46-55)

%% Project Financial Parameters (项目财务参数)
project_lifetime  = 20;    % Project lifetime (years) 项目寿命
discount_rate     = 0.08;  % Annual discount rate (折现率) 8%

%% Initial Capital Expenditure — CAPEX (初始资本支出)
% Photovoltaic system: 1 MWp @ $1 000/kWp installed
capex_pv         = 1000000;  % $ 光伏系统

% Energy storage: 2 MWh Li-ion @ $300/kWh + $100/kW inverter
capex_es         =  700000;  % $ 储能系统 (2 MWh × $300/kWh + 1 MW × $100/kW)

% V2G infrastructure upgrade for 40 EVs (chargers, metering, aggregation)
capex_ev_infra   =  200000;  % $ 电动汽车 V2G 基础设施

% Smart thermostat + control hardware for 3 TCL buildings
capex_tcl        =   30000;  % $ 温控负荷控制设备

% Smart control integration for 10 IPP processes
capex_ipp        =  100000;  % $ 工业生产过程控制集成

% VPP management platform: software, communication, SCADA
capex_vpp_system =  150000;  % $ 虚拟电厂管理系统

% Total initial CAPEX (总初始投资)
capex_total = capex_pv + capex_es + capex_ev_infra + ...
              capex_tcl + capex_ipp + capex_vpp_system;

%% Battery Replacement Costs (电池替换成本)
% ES replacement at end-of-life (technology learning curve reduces cost)
capex_es_replace = 450000;  % $ — reduced cost at ~year 10-12

%% Capacity Degradation Parameters (容量衰减参数)
% Two-factor degradation model (two-factor model for Li-ion batteries):
%   Q(k) = max( Q0 * (1 - k*alpha - N_cyc(k)*beta),  Q_eol )
% where k = years elapsed, N_cyc = equivalent full cycles at year k,
%       alpha = annual calendar coefficient, beta = per-cycle coefficient.

% --- Energy Storage (ES) — NMC Li-ion ---
es_calendar_aging_rate = 0.020;   % 2.0%/year calendar degradation 日历老化率
es_cycle_aging_factor  = 0.0001;  % 0.01% per equivalent full cycle 循环老化系数
es_capacity_mwh        = 2.0;     % Nominal capacity (MWh) 标称容量
es_eol_threshold       = 0.80;    % End-of-life at 80% remaining 寿命终止阈值

% --- Electric Vehicles (EV) — NMC/LFP mix, 40 × 60 kWh ---
ev_calendar_aging_rate = 0.025;   % 2.5%/year calendar degradation
ev_cycle_aging_factor  = 0.00015; % 0.015% per equivalent full cycle
ev_capacity_kwh        = 60;      % Per-vehicle nominal capacity (kWh)
ev_count               = 40;      % Number of EVs in the fleet
ev_eol_threshold       = 0.80;    % End-of-life threshold

% --- Photovoltaic System (PV) ---
pv_annual_degradation  = 0.005;   % 0.5%/year linear degradation 光伏年衰减率

%% Annual Fixed O&M Costs (年度固定运维费用)
opex_es_annual        = 10000;  % $ ES maintenance ($5/kWh·yr)
opex_pv_annual        = 10000;  % $ PV maintenance ($10/kWp·yr)
opex_ev_infra_annual  =  5000;  % $ V2G infrastructure maintenance
opex_tcl_annual       =  3000;  % $ TCL control system maintenance
opex_ipp_annual       =  5000;  % $ IPP control system maintenance
opex_vpp_annual       = 15000;  % $ VPP platform license and operation

% Total annual fixed O&M (年度固定运维总费用)
opex_fixed_annual = opex_es_annual + opex_pv_annual + opex_ev_infra_annual + ...
                    opex_tcl_annual + opex_ipp_annual + opex_vpp_annual;

%% Revenue Scaling (收益季节性修正)
% Simulation data are from July 2020 (PJM summer peak for frequency
% regulation).  Seasonal factors scale July revenue to each calendar
% month so that a full-year estimate can be derived.
% Basis: 1.0 = July (reference month).
seasonal_factors = [0.90, 0.90, 0.85, 0.85, 0.85, 0.95, ...  % Jan-Jun
                    1.00, 0.95, 0.85, 0.85, 0.85, 0.90];      % Jul-Dec

% Average seasonal correction factor (average of 12 months / July)
annual_scaling = sum(seasonal_factors) / 12;   % ~0.896

% Conservative revenue growth assumption (no escalation)
revenue_growth_rate = 0.00;   % 0% per year (保守假设：无市场增长)

%% Scenario Analysis Ranges (情景分析参数范围)
% Discount rate scenarios: optimistic / base / conservative
discount_rate_scenarios   = [0.05, 0.08, 0.10];

% Degradation rate multipliers: slow / base / fast aging
degradation_scenarios     = [0.5, 1.0, 2.0];

%% Derived Cost Parameters (from data_prepare_parameters.m)
pr_dis_es = 100;   % ES discharge degradation cost ($/MWh)
pr_ch_es  = 100;   % ES charge degradation cost ($/MWh)
pr_dis_ev = 150;   % EV discharge degradation cost ($/MWh)
pr_ch_ev  =   0;   % EV charge degradation cost ($/MWh) — zero
