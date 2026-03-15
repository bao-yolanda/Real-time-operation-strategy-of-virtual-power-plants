%% Visualisation of Lifecycle Investment Assessment Results
% 全生命周期投资评估结果可视化
%
% Generates a 2×3 figure with:
%   (1,1) Capacity degradation curves for ES, EV, PV, and combined
%   (1,2) Annual revenue vs. total O&M cost (grouped bar chart)
%   (1,3) Annual net cash flow (bar chart)
%   (2,1) Cumulative cash flow with payback period markers
%   (2,2) Lifetime revenue / cost breakdown (stacked bar by resource)
%   (2,3) Key financial metrics summary (text panel)
%
% This script can be run standalone after loading results, or is called
% automatically from lifecycle_assessment_main.m.

%% -------------------------------------------------------------------------
%  Load results if not already in workspace
%  -------------------------------------------------------------------------
if ~exist('lifecycle_result', 'var')
    load(fullfile('..', 'results', 'lifecycle_assessment_result.mat'));
end

% Unpack for convenience
proj_lt         = lifecycle_result.project_lifetime;
dr              = lifecycle_result.discount_rate;
years           = lifecycle_result.years;
es_cap          = lifecycle_result.es_cap;
ev_cap          = lifecycle_result.ev_cap;
pv_cap          = lifecycle_result.pv_cap;
capacity        = lifecycle_result.capacity;
annual_revenue  = lifecycle_result.annual_revenue;
annual_opex     = lifecycle_result.annual_total_opex;
annual_cf       = lifecycle_result.annual_cashflow;
cum_cf          = lifecycle_result.cumulative_cashflow;
cum_dcf         = lifecycle_result.cumulative_discounted;
npv             = lifecycle_result.npv;
irr             = lifecycle_result.irr;
simple_pbp      = lifecycle_result.simple_pbp;
dynamic_pbp     = lifecycle_result.dynamic_pbp;
capex_total     = lifecycle_result.capex_total;
es_rep_yr       = lifecycle_result.es_replacement_years;

yr_axis = 1 : proj_lt;   % x-axis for annual quantities

%% -------------------------------------------------------------------------
%  Figure Setup
%  -------------------------------------------------------------------------
figure('Name', 'Lifecycle Investment Assessment / 全生命周期投资评估', ...
       'NumberTitle', 'off');
figW = 40;  figH = 26;
set(gcf, 'Units', 'centimeters', 'Position', [2 2 figW figH]);

font_name = 'Times New Roman';
font_size = 11;
lw        = 1.5;

%% -------------------------------------------------------------------------
%  Subplot 1 — Capacity Degradation Curves (容量衰减曲线)
%  -------------------------------------------------------------------------
ax1 = subplot(2, 3, 1);
hold(ax1, 'on');

plot(years, es_cap * 100, '-bs', 'LineWidth', lw, 'MarkerSize', 4, ...
     'DisplayName', '储能系统 (ES)');
plot(years, ev_cap * 100, '-ro', 'LineWidth', lw, 'MarkerSize', 4, ...
     'DisplayName', '电动汽车 (EV)');
plot(years, pv_cap * 100, '-g^', 'LineWidth', lw, 'MarkerSize', 4, ...
     'DisplayName', '光伏系统 (PV)');
plot(years, capacity * 100, '-k',  'LineWidth', 2, ...
     'DisplayName', '综合加权容量');
yline(ax1, 80, '--m', 'LineWidth', 1);
text(0.5, 81, 'EOL (80%)', 'FontSize', 9, 'FontName', font_name, ...
     'Color', 'm');

% Mark all ES replacement years
for rep_yr = es_rep_yr
    if rep_yr <= proj_lt
        xline(ax1, rep_yr, '-.b', 'LineWidth', 1);
        text(rep_yr + 0.1, 96, sprintf('↑yr%d', rep_yr), 'FontSize', 8, ...
             'FontName', font_name, 'Color', 'b');
    end
end
if ~isempty(es_rep_yr)
    text(proj_lt*0.02, 97, 'ES更换年', 'FontSize', 8, 'FontName', font_name, ...
         'Color', 'b');
end

hold(ax1, 'off');
xlabel(ax1, '年份 (Year)', 'FontSize', font_size, 'FontName', font_name);
ylabel(ax1, '容量保留率 (%)', 'FontSize', font_size, 'FontName', font_name);
title(ax1, '设备容量衰减曲线', 'FontSize', font_size+1, 'FontName', font_name);
legend(ax1, 'Location', 'SouthWest', 'FontSize', 9, 'FontName', font_name);
xlim(ax1, [0, proj_lt]);
ylim(ax1, [60, 105]);
grid(ax1, 'on');
ax1.FontName = font_name;
ax1.FontSize = 9;

%% -------------------------------------------------------------------------
%  Subplot 2 — Annual Revenue vs. O&M Cost (年度收益与运营成本)
%  -------------------------------------------------------------------------
ax2 = subplot(2, 3, 2);
bar_data = [annual_revenue', annual_opex'];
b = bar(ax2, yr_axis, bar_data, 'grouped');
b(1).FaceColor = [0.20, 0.60, 0.90];
b(2).FaceColor = [0.90, 0.30, 0.30];
xlabel(ax2, '年份 (Year)', 'FontSize', font_size, 'FontName', font_name);
ylabel(ax2, '金额 ($)',    'FontSize', font_size, 'FontName', font_name);
title(ax2, '年度净收益与运营成本', 'FontSize', font_size+1, 'FontName', font_name);
legend(ax2, {'年度净收益', '年度总运营成本'}, ...
       'Location', 'NorthEast', 'FontSize', 9, 'FontName', font_name);
grid(ax2, 'on');
ax2.FontName = font_name;
ax2.FontSize = 9;

%% -------------------------------------------------------------------------
%  Subplot 3 — Annual Net Cash Flow (年度净现金流)
%  -------------------------------------------------------------------------
ax3 = subplot(2, 3, 3);
cf_colors = repmat([0.20, 0.70, 0.30], proj_lt, 1);
cf_colors(annual_cf < 0, :) = repmat([0.85, 0.25, 0.25], ...
                               sum(annual_cf < 0), 1);
b3 = bar(ax3, yr_axis, annual_cf, 'FaceColor', 'flat');
b3.CData = cf_colors;
yline(ax3, 0, 'k-', 'LineWidth', 1);
xlabel(ax3, '年份 (Year)', 'FontSize', font_size, 'FontName', font_name);
ylabel(ax3, '现金流 ($)',  'FontSize', font_size, 'FontName', font_name);
title(ax3, '年度净现金流', 'FontSize', font_size+1, 'FontName', font_name);
grid(ax3, 'on');
ax3.FontName = font_name;
ax3.FontSize = 9;

%% -------------------------------------------------------------------------
%  Subplot 4 — Cumulative Cash Flow & Payback Periods (累计现金流与回收期)
%  -------------------------------------------------------------------------
ax4 = subplot(2, 3, 4);
hold(ax4, 'on');
plot(ax4, yr_axis, cum_cf,  '-bs', 'LineWidth', lw, 'MarkerSize', 4, ...
     'DisplayName', '未折现累计现金流');
plot(ax4, yr_axis, cum_dcf, '-ro', 'LineWidth', lw, 'MarkerSize', 4, ...
     'DisplayName', sprintf('折现累计现金流 (r=%.0f%%)', dr*100));
yline(ax4, 0, 'k--', 'LineWidth', 1.2);

if isfinite(simple_pbp) && simple_pbp <= proj_lt
    xline(ax4, simple_pbp, '--b', 'LineWidth', 1);
    text(simple_pbp + 0.2, min(cum_cf)*0.1, ...
         sprintf('静态PP: %.1f年', simple_pbp), ...
         'FontSize', 9, 'FontName', font_name, 'Color', 'b');
end
if isfinite(dynamic_pbp) && dynamic_pbp <= proj_lt
    xline(ax4, dynamic_pbp, '--r', 'LineWidth', 1);
    text(dynamic_pbp + 0.2, min(cum_dcf)*0.1, ...
         sprintf('动态PP: %.1f年', dynamic_pbp), ...
         'FontSize', 9, 'FontName', font_name, 'Color', 'r');
end

hold(ax4, 'off');
xlabel(ax4, '年份 (Year)',  'FontSize', font_size, 'FontName', font_name);
ylabel(ax4, '累计现金流 ($)', 'FontSize', font_size, 'FontName', font_name);
title(ax4, '累计现金流（含回收期标记）', ...
      'FontSize', font_size+1, 'FontName', font_name);
legend(ax4, 'Location', 'NorthWest', 'FontSize', 9, 'FontName', font_name);
grid(ax4, 'on');
ax4.FontName = font_name;
ax4.FontSize = 9;

%% -------------------------------------------------------------------------
%  Subplot 5 — Revenue Composition Over Lifetime (全生命周期收益构成)
%  -------------------------------------------------------------------------
ax5 = subplot(2, 3, 5);
total_rev   = sum(annual_revenue);
total_opex_sum  = sum(annual_opex);
total_capex_rep = sum(lifecycle_result.annual_capex_replacement);
net_profit  = total_rev - total_opex_sum - total_capex_rep - capex_total;

values     = [capex_total, total_capex_rep, total_opex_sum, max(net_profit,0)];
colors_pie = [0.85 0.35 0.35; 0.95 0.65 0.15; 0.45 0.65 0.85; 0.35 0.75 0.45];

bar_h = bar(ax5, 1:4, values, 'FaceColor', 'flat');
for k = 1:4
    bar_h.CData(k,:) = colors_pie(k,:);
end
set(ax5, 'XTickLabel', {'初始投资', '储能替换', '累计运营成本', '累计净利润'});
xtickangle(ax5, 15);
ylabel(ax5, '金额 ($)', 'FontSize', font_size, 'FontName', font_name);
title(ax5, sprintf('全生命周期收益构成 (%d年)', proj_lt), ...
      'FontSize', font_size+1, 'FontName', font_name);
grid(ax5, 'on');
ax5.FontName = font_name;
ax5.FontSize = 9;

%% -------------------------------------------------------------------------
%  Subplot 6 — Financial Metrics Summary (财务指标汇总文字面板)
%  -------------------------------------------------------------------------
ax6 = subplot(2, 3, 6);
axis(ax6, 'off');

if ~isnan(irr)
    irr_str = sprintf('%.2f%%', irr * 100);
else
    irr_str = '不可计算';
end
if isfinite(simple_pbp)
    spbp_str = sprintf('%.1f 年', simple_pbp);
else
    spbp_str = sprintf('>%d 年', proj_lt);
end
if isfinite(dynamic_pbp)
    dpbp_str = sprintf('%.1f 年', dynamic_pbp);
else
    dpbp_str = sprintf('>%d 年', proj_lt);
end

% Capacity at representative years (bounds-safe; capacity is indexed 1..proj_lt+1)
n_cap = length(capacity);
cap5  = capacity(min(6,        n_cap));   % year  5
cap10 = capacity(min(11,       n_cap));   % year 10
cap15 = capacity(min(16,       n_cap));   % year 15
cap20 = capacity(min(proj_lt+1, n_cap));  % final project year

lines = {
    '■ 全生命周期投资评估摘要', ...
    '', ...
    sprintf('项目寿命:          %d 年', proj_lt), ...
    sprintf('折现率:            %.1f%%', dr*100), ...
    '', ...
    '— 初始总投资 (CAPEX) —', ...
    sprintf('  $%d', capex_total), ...
    '', ...
    '— 年度基准净现金流 (第1年) —', ...
    sprintf('  $%.0f', annual_cf(1)), ...
    '', ...
    '— 财务指标 —', ...
    sprintf('  NPV: $%.0f', npv), ...
    sprintf('  IRR: %s', irr_str), ...
    sprintf('  静态回收期: %s', spbp_str), ...
    sprintf('  动态回收期: %s', dpbp_str), ...
    '', ...
    '— 综合容量 (第5/10/15/20年) —', ...
    sprintf('  %.1f%% / %.1f%% / %.1f%% / %.1f%%', ...
            cap5*100, cap10*100, cap15*100, cap20*100), ...
};

line_h = 0.047;
for i = 1 : length(lines)
    text(ax6, 0.03, 1 - (i-1)*line_h, lines{i}, ...
         'Units', 'normalized', ...
         'FontSize', 9.5, ...
         'FontName', font_name, ...
         'VerticalAlignment', 'top');
end

%% -------------------------------------------------------------------------
%  Save Figure  (保存图表)
%  -------------------------------------------------------------------------
set(gcf, 'PaperSize', [figW, figH]);
saveas(gcf, fullfile('..', 'results', 'lifecycle_assessment_plot.pdf'));
saveas(gcf, fullfile('..', 'results', 'lifecycle_assessment_plot.png'));
fprintf('生命周期评估图表已保存至 ../results/\n');
