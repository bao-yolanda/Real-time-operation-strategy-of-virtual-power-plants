function irr = compute_irr(initial_investment, annual_cashflows)
%% Compute Internal Rate of Return (IRR)
% 计算内部收益率 (IRR)
%
% Uses the Newton-Raphson method to find the discount rate r that makes
% the Net Present Value (NPV) of the cash-flow stream equal to zero:
%
%   NPV(r) = -C0 + CF1/(1+r)^1 + CF2/(1+r)^2 + ... + CFn/(1+r)^n = 0
%
% Inputs
%   initial_investment  – initial outlay (positive value; sign handled
%                         internally so pass the absolute CAPEX figure)
%   annual_cashflows    – 1×N vector of annual net cash flows (years 1–N)
%
% Output
%   irr  – internal rate of return as a decimal (e.g. 0.12 = 12%)
%          Returns NaN if the method does not converge or if the project
%          cash flows do not support a positive IRR.

cashflows = [-abs(initial_investment), annual_cashflows];
n_cf      = length(cashflows);
yr        = 0 : n_cf - 1;   % Year indices 0, 1, 2, …, n

% --- Initial guess ---
% Use a heuristic: ratio of total undiscounted returns to investment.
total_return = sum(annual_cashflows);
if total_return <= 0
    % Project never returns capital — IRR is undefined / negative
    irr = NaN;
    return;
end
r = total_return / (abs(initial_investment) * n_cf);
r = max(0.001, min(r, 0.50));   % Clamp to [0.1%, 50%] as starting point

max_iter = 2000;
tol      = 1e-10;

for iter = 1 : max_iter
    discount = (1 + r) .^ yr;
    npv_val  = sum(cashflows ./ discount);
    dnpv_dr  = -sum(yr .* cashflows ./ ((1 + r) .^ (yr + 1)));

    if abs(dnpv_dr) < eps
        break;
    end

    r_new = r - npv_val / dnpv_dr;

    % Clamp to avoid divergence
    r_new = max(-0.99, min(r_new, 10.0));

    if abs(r_new - r) < tol
        r = r_new;
        break;
    end
    r = r_new;
end

% Final NPV check — if residual is large the solver did not converge
discount = (1 + r) .^ yr;
residual = abs(sum(cashflows ./ discount));
if residual > 1.0   % $1 tolerance on a project of millions
    irr = NaN;
else
    irr = r;
end
end
