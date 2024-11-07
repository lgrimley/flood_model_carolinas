clear
clc

% Data I am reading into this script was downloaded from globus in
% /GCM_surge/ADCIRC/ncep/StormTide/


% load the ADCIRC x,y locations
load 'C:\Users\lelise\Documents\globus_data\coastline_lores'

% load the TC track file 
load 'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'

% change to directory where m_lldist_L script is located
cd Z:\Data-Expansion\users\lelise\Chapter3\scripts\wind

% X,Y location of the ADCIRC gage at the Cape Fear outlet (Gage 194)
clat = yvec(194); 
clong = xvec(194);
select_tcs = [];
not_selected_tcs = [];
% Select the TCs whose tracks are within 200km of this point
for i = 1:5018
    tr_lat = lat100(i,:); 
    tr_lat = tr_lat(tr_lat~=0); 
    tr_lon = lon100(i,:); 
    tr_lon = tr_lon(1:length(tr_lat)); 
    trk = length(tr_lat); % track points

    % You can comment this out if you already selected the tracks you want
    % to model, otherwise this searches for storms within a distance of the
    % center of the study area
    d = []; 
    for j = 1:trk
        [d(j),~,~] = m_lldist_L([tr_lon(j), clong], [tr_lat(j), clat]);
    end
    if min(d)>200
        not_selected_tcs(end+1) = i;
        continue
    end
    select_tcs(end+1) = i;

end

% Now load the matlab data saved from the ADCIRC simulations -- this loads
%
% WL_inds
% - each row corresponds to an ADCIRC gage output (e.g., 355 stations)
% - the columns list the Storm IDs that are relevant for that gage location
% 
% WL_mat
% - shape(355, 2000, 447) = (station, storms, ADCIRC water level output timesteps)
% - the rows are timeseries for a single TC storm at that gage
% - each row corresponds to the Storm ID in the WL_inds data (e.g., row 1 is Storm ID in column 1)

load 'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\ncep_reanal_WLseries.mat'

% If indexing above is correct, the storms saved in the WL_inds should
% match (at least mostly) the storms selected with 200 km of the points
% above at the ADCIRC station 194
WL_inds = WL_inds_adj;
gage_194 = WL_inds(194,:);

% Find common values
[commonValues, idx1, idx2] = intersect(select_tcs, gage_194);

