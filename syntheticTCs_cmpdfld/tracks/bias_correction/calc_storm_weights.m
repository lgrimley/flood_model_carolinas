clear
clc
addpath 'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld\tracks\bias_correction'
cd Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\BiasCorrection

% choose model 
model_map = containers.Map({1, 2, 3, 4, 5}, {'canesm', 'cnrm6', ...
    'ecearth6', 'ipsl6', 'miroc6'});

%rps = 5:500; 

% Read in the historical (NCEP) vmax at landfall
temp = readtable("vmax/NCEP/ncep_landfall_vmax.csv");
xoh = temp.vstore100; 
xoh_tot = length(xoh); 
% Fit a distribution to the TC vmax
[xoh_fit, ~, oh_type] = fit_gpd(xoh, 15, 4);
xcomp = min(xoh):1:max(xoh)*1.2; 


iter = 1;
G_comp = [];
total = 0;
st_weights_hist = {};
st_weights_proj = {};

% for each model
% currently set to only do one GCM 

for i = 1:length(model_map)
    m = i; 
    %name = strcat(model_map(i), '_20thcal_TCchars'); 

    % load historical GCM track characteristics
    fname = sprintf('vmax/CMIP6_20thcal/UScoast6_AL_%s_20thcal_roEst1rmEst1_trk100.mat_select_vstore100.csv', ...
        model_map(i));
    temp = readtable(fname);
    xmh = temp.vstore100; % xmh is modeled historical (from GCM)
    inds = temp.tc_id;
    xmh_tot = length(xmh);
    if xmh_tot < 50
        continue
    end
    %  fit distribution of modeled historical 
    [xmh_fit, ~, mh_type] = fit_gpd(xmh, 15, 4); 

    % load future GCM track characteristics
    fname = sprintf('vmax/CMIP6_585/%s_ssp585_landfall_vmax.csv', ...
        model_map(i));
    temp = readtable(fname);
    tc_ids = temp.tc_id;
    xmp = temp.vstore100; % xmp is model projected (future GCM)
    xmp_tot = length(xmp); 
    % fit modeled projected distribution
    [xmp_fit, ~, mp_type] = fit_gpd(xmp, 15, 4); 


    % QDM is quantile delta mapping to get biases at each quantile of
    % the distribution where:
    % xphat is the bias corrected for the modeled future
    % xhhat is the bias corrected for the modeled historical
    [xphat, xhhat] = QDM(xoh, xmh, xmp, 4);

    % Now fit a distribution for the modeled future using the bias
    % corrected data
    [xphat_fit, ~, phat_type] = fit_gpd(xphat, 15, 4); 

    % this if loop calculates the weights of each storm for the historical 
    % GCM tracks based on the difference between the pdf of xoh (NCEP historical)
    % and pdf of xmh (GCM historical)
    if strcmp(oh_type,'kde') || strcmp(mh_type,'kde')
        bw = (max(xmh)-min(xmh))/12; % minimum and maximum Vmax values for historical -- I think this is the bin width with 12 being arbitrary or quantiles I think
        edges = 0:round(bw):max(xmh)+round(bw); % edge of the bins
        c_mh = histcounts(xmh, 'BinEdges', edges, 'Normalization', ...
            'probability'); % histogram for modeled historical
        c_oh = histcounts(xoh, 'BinEdges', edges, 'Normalization', ...
            'probability'); % histogram for NCEP historical
        tau = 999*ones(length(c_oh),1); % create an empty array for saving taus
        tau(c_mh~=0)  = c_oh(c_mh~=0)./c_mh(c_mh~=0); % calculate the ratio of NCEP to modeled historical
        tau(tau==999) = mean(tau(c_mh~=0)); % if there are gaps, set them to the mean ratio
        weight = tau./sum(tau); % now normalize the weights based on their sum 
        wi_h = zeros(xmh_tot, 1);
        % now loop through the modeled historical storms (in the order they
        % were read in from the CSV, e.g., not sorted)
        for j = 1:xmh_tot
            ind = find(xmh(j)<edges, 1); % this finds what Vmax bin the storm is located in 
            wi_h(j) = weight(max(ind-1,1)); % this returns the weight for that storm
        end
        wi_h = wi_h./sum(wi_h);
    else
        tau = pdf(xoh_fit, xmh)./pdf(xmh_fit, xmh);
        tau(tau==0) = min(tau(tau>0));
        wi_h = tau./sum(tau);
    end

    % this if loop calculates the weights of each storm for the future
    % GCM tracks based on the difference between the pdf of xphat (bias 
    % corrected GCM future) and the pdf of xmp (biased GCM future)
    st_weights_hist{iter} = wi_h;
    %phat_type = 'kde';
    if strcmp(mp_type,'kde') || strcmp(phat_type,'kde')
        bw = (max(xmp)-min(xmp))/12; 
        edges = 0:round(bw):max(xmp)+round(bw); 
        c_mp = histcounts(xmp, 'BinEdges', edges, 'Normalization', ...
            'probability'); 
        c_phat = histcounts(xphat, 'BinEdges', edges, 'Normalization', ...
            'probability');
        tau = 999*ones(length(c_phat),1);
        tau(c_mp~=0)  = c_phat(c_mp~=0)./c_mp(c_mp~=0);
        tau(tau==999) = mean(tau(c_mp~=0));
        weight = tau./sum(tau);
        wi_p = zeros(xmp_tot, 1);
        for j = 1:xmp_tot
            ind = find(xmp(j)<edges, 1); 
            wi_p(j) = weight(max(ind-1,1)); 
        end
        wi_p(wi_p == 0) = 1*10^-5; 
        wi_p = wi_p./sum(wi_p);
    else
        tau = pdf(xphat_fit, xmp)./pdf(xmp_fit, xmp); 
        tau(tau==0) = min(tau(tau>0));
        wi_p = tau./sum(tau);
    end
    st_weights_proj{iter} = wi_p; 
    out = [tc_ids, xmp, wi_p];
    fname = sprintf('%s_ssp585_weighted.csv', ...
        model_map(i));
    writematrix(out, fname);
    iter = iter + 1;
end
    
%% The storm weights should correspond to the order of the TC IDs and 
% Vmax that was read into the script at the beginning
save('storm_weights.mat', 'st_weights_hist', 'st_weights_proj')
