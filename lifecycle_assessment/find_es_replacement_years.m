function rep_years = find_es_replacement_years(project_lifetime, es_annual_efcs, ...
                                                 calendar_rate, cycle_factor, eol_threshold)
%FIND_ES_REPLACEMENT_YEARS  Find all ES battery replacement years.
%
%   rep_years = FIND_ES_REPLACEMENT_YEARS(project_lifetime, es_annual_efcs,
%               calendar_rate, cycle_factor, eol_threshold)
%
%   Iteratively locates every year within [1, project_lifetime-1] at which
%   the energy-storage battery capacity reaches eol_threshold (80%).  After
%   each replacement the battery resets to 100% capacity and ages again at
%   the same rate.  Replacements in the final project year are excluded
%   because they would cost capex_es_replace with zero remaining useful life.
%
%   Inputs
%     project_lifetime  – total project duration (integer, years)
%     es_annual_efcs    – equivalent full cycles per year for the ES
%     calendar_rate     – annual calendar aging fraction (e.g. 0.020)
%     cycle_factor      – capacity loss per equivalent full cycle (e.g. 0.0001)
%     eol_threshold     – end-of-life capacity fraction (e.g. 0.80)
%
%   Output
%     rep_years  – row vector of replacement years (may be empty)

rep_years = [];
last_rep  = 0;   % Year of the most-recent replacement (0 = never replaced)

while true
    % Find next year when capacity drops to eol_threshold.
    % Cap at project_lifetime-1 to avoid pointless last-year replacement.
    next_eol = -1;
    for yr = (last_rep + 1) : (project_lifetime - 1)
        k_since = yr - last_rep;
        n_cyc   = k_since * es_annual_efcs;
        cap_est = 1 - k_since * calendar_rate - n_cyc * cycle_factor;
        if cap_est <= eol_threshold
            next_eol = yr;
            break;
        end
    end

    if next_eol < 0 || next_eol >= project_lifetime
        break;   % Survives to end (or only hits EOL in last year)
    end

    rep_years(end + 1) = next_eol; %#ok<AGROW>
    last_rep = next_eol;
end
end
