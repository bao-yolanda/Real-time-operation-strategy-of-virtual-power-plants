function [capacity, es_cap, ev_cap, pv_cap] = ...
        capacity_degradation_model(years, daily_es_throughput_mwh, ...
                                   daily_ev_throughput_kwh, params)
%% Capacity Degradation Model for VPP Resources
% 虚拟电厂资源容量衰减模型
%
% Implements a two-factor empirical degradation model for battery-type
% resources (ES and EV) and a linear degradation model for PV.
%
% Two-factor battery degradation:
%   Q(k) = max( Q0 * (1 - k*alpha - N_cyc(k)*beta),  Q_eol )
%   where k        = years elapsed
%         N_cyc(k) = equivalent full cycles accumulated by year k
%         alpha    = annual calendar aging coefficient
%         beta     = per-cycle aging coefficient
%         Q_eol    = end-of-life capacity threshold
%
% PV linear degradation:
%   Q_pv(k) = max( 1 - k * gamma_pv,  0 )
%
% Inputs
%   years                  – row vector of year indices, e.g. 0:20
%   daily_es_throughput_mwh – average daily energy throughput of ES (MWh),
%                             used to calculate equivalent full cycles
%   daily_ev_throughput_kwh – average daily energy throughput per EV (kWh)
%   params                 – struct with fields:
%       .es_calendar_aging_rate  (alpha_es)
%       .es_cycle_aging_factor   (beta_es)
%       .es_capacity_mwh         (ES nominal capacity in MWh)
%       .es_eol_threshold
%       .ev_calendar_aging_rate  (alpha_ev)
%       .ev_cycle_aging_factor   (beta_ev)
%       .ev_capacity_kwh         (per-vehicle nominal capacity in kWh)
%       .ev_eol_threshold
%       .pv_annual_degradation   (gamma_pv)
%
% Outputs
%   capacity  – overall normalised capacity (weighted combination, 0-1)
%   es_cap    – ES remaining capacity ratio, same length as years
%   ev_cap    – EV fleet remaining capacity ratio
%   pv_cap    – PV remaining capacity ratio
%
% Capacity weights (based on rated regulation power of each resource type):
%   ES  = 1.00 MW  → weight 0.30
%   EV  = 0.88 MW  → weight 0.28  (40 × 22 kW = 880 kW)
%   PV  = 1.00 MW  → weight 0.22  (daytime average)
%   TCL + IPP      → weight 0.20  (no significant degradation)

n_years = length(years);
es_cap  = zeros(1, n_years);
ev_cap  = zeros(1, n_years);
pv_cap  = zeros(1, n_years);

% Annual equivalent full cycles (each full cycle = 1 discharge + 1 charge)
es_annual_efcs = daily_es_throughput_mwh * 365 / params.es_capacity_mwh / 2;
ev_annual_efcs = daily_ev_throughput_kwh * 365 / params.ev_capacity_kwh  / 2;

for i = 1 : n_years
    k = years(i);

    % --- ES capacity (calendar + cycle aging) ---
    n_cyc_es  = k * es_annual_efcs;
    es_cap(i) = max(1 - k * params.es_calendar_aging_rate ...
                      - n_cyc_es * params.es_cycle_aging_factor, ...
                    params.es_eol_threshold);

    % --- EV fleet capacity (calendar + cycle aging) ---
    n_cyc_ev  = k * ev_annual_efcs;
    ev_cap(i) = max(1 - k * params.ev_calendar_aging_rate ...
                      - n_cyc_ev * params.ev_cycle_aging_factor, ...
                    params.ev_eol_threshold);

    % --- PV capacity (linear degradation) ---
    pv_cap(i) = max(1 - k * params.pv_annual_degradation, 0);
end

% Weighted overall capacity (weights proportional to rated regulation power)
w_es    = 0.30;
w_ev    = 0.28;
w_pv    = 0.22;
w_other = 0.20;   % TCL + IPP — no significant capacity degradation

capacity = w_es * es_cap + w_ev * ev_cap + w_pv * pv_cap ...
         + w_other * ones(1, n_years);
end
