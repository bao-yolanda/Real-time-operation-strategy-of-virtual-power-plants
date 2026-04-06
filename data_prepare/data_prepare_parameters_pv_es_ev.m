%% Read only PV, ES, EV parameters (NO IPP and TCL)
% All need to be converted to MW units

%% New Energy (PV)
load("output_pv.mat");
param.power_dis_upper_limit_pv = output_pv';
param.power_dis_lower_limit_pv = 0 * output_pv';

%% Energy Storage
param.energy_init_es = 1;
param.energy_upper_limit_es = 0.9 * 2;
param.energy_lower_limit_es = 0.1 * 2;
param.power_dis_upper_limit_es = 1;
param.power_dis_lower_limit_es = 0;
param.power_ch_upper_limit_es = 1;
param.power_ch_lower_limit_es = 0;
param.theta_es = 1;
param.eta_dis_es = 0.90;
param.eta_ch_es = 0.90;
param.pr_dis_es = 100;
param.pr_ch_es = 100;

%% Electric Vehicles
filename = 'EV_arrive_leave.xlsx';
sheet = 'EV_arrive_leave';
xlRange = 'A2:C121';
EV_arrive_leave = xlsread(filename, sheet, xlRange);

param.energy_init_ev = 20 * 1e-3;
param.energy_end_ev = 50 * 1e-3;
param.energy_upper_limit_ev = 60 * 0.9 * 1e-3;
param.energy_lower_limit_ev = 60 * 0.1 * 1e-3;
param.power_dis_upper_limit_ev = 22 * 1e-3;
param.power_dis_lower_limit_ev = 0;
param.power_ch_upper_limit_ev = 22 * 1e-3;
param.power_ch_lower_limit_ev = 0;
param.theta_ev = 1;
param.eta_dis_ev = 0.90;
param.eta_ch_ev = 0.90;
param.pr_dis_ev = 150;
param.pr_ch_ev = 0;

% Number of EVs
NOFEV = length(EV_arrive_leave);

% Charging status u
param.u = zeros(NOFEV, NOFSLOTS);
for idx = 1 : NOFEV
    for jdx = 1 : NOFSLOTS
        if EV_arrive_leave(idx, 2) <= jdx && jdx <= EV_arrive_leave(idx, 3)
            param.u(idx, jdx) = 1;
        end
    end
end

clear EV_arrive_leave idx jdx
