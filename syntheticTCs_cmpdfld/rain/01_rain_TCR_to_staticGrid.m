clear
clc
% script for post-processing TCRM rainfall into static user-defined grid

% USER INPUT 
% bb: [LonMin, LatMin; 
%      LonMax, LatMax] boundingbox 
% lat_inc : latitude grid increment
% lon_inc : longitude grid increment
% rain_folder: location with all TCRM matfiles 
% outfolder: location where outputs will be stored
% Ns = number of storms 

% OUTPUTS 
% rain files: one mat file containing area-averaged time series of each 
% storm and maximum rain time series of each storm.

% specify bounding box
bb = [-83.67,31.95;-75.11,36.81];  % SFINCS domain of NC/SC
gcm = {'canesm_ssp585'; 'cnrm6_ssp585'; 'ecearth6_ssp585'; 'ipsl6_ssp585'; 'miroc6_ssp585'}; 
cd 'Z:\Data-Expansion\users\lelise\Chapter3\CMIP6_585';

for kk = 1:length(gcm)
    %trackpath = fullfile('.\tracks', strcat('UScoast6_AL_', gcm{kk}, 'cal_roEst1rmEst1_trk100.mat'));
    inputfolder = fullfile('.\rain', strcat(gcm{kk}, 'cal'));
    outputfolder = fullfile('.\rain', strcat('TCR_Gridded_' ,gcm{kk}, 'cal'));
    selected_tracks = readtable(fullfile('.\stormTide', strcat('stormTide_TCIDs_and_gageCounts_', gcm{kk}, '.csv')));
    
    if ~isfolder(outputfolder)
        mkdir(outputfolder);
    end

    % create global grid 
    lat_inc = 0.05; 
    lon_inc = 0.05;
    xvec = floor(bb(1,1)):lon_inc:ceil(bb(2,1)); 
    yvec = floor(bb(1,2)):lat_inc:ceil(bb(2,2)); 
    min_lon = xvec(1);
    min_lat = yvec(1);
    max_lon = xvec(end);
    max_lat = yvec(end);
    
    % loop through each storm 
    for ii = 1:length(selected_tracks.tc_id)
        % load rainfall mat file 
        stnum = selected_tracks.tc_id(ii);
    
        % Create a new NetCDF files
        if stnum < 10
            stormfile = strcat('000',num2str(stnum),'.nc');
        elseif stnum >= 10 && stnum < 100
            stormfile = strcat('00', num2str(stnum), '.nc');
        elseif stnum >= 100 && stnum < 1000
            stormfile = strcat('0', num2str(stnum), '.nc');
        else
            stormfile = strcat(num2str(stnum), '.nc');
        end 
        f = sprintf('%s%s',outputfolder,stormfile);
    
        % If the file already exists, this TC has been processed already
        if isfile(f) == 0
            disp(stnum)
            % Read the TCR output for the storm
            % This loads PLAT_SAVE, PLONG_SAVE, rain
            if stnum < 10
                stormfile = strcat('000',num2str(stnum),'.mat');
            elseif stnum >= 10 && stnum < 100
                stormfile = strcat('00', num2str(stnum), '.mat');
            elseif stnum >= 100 && stnum < 1000
                stormfile = strcat('0', num2str(stnum), '.mat');
            else
                stormfile = strcat(num2str(stnum), '.mat');
            end 
            filename = fullfile(inputfolder,stormfile);
            load(filename)
        
            % Convert longitude from 360 degrees to 180
            PLONG_SAVE(PLONG_SAVE>180) = PLONG_SAVE(PLONG_SAVE>180)-360;
            [~, ~, trk] = size(rain);
            tstart = 4;
        
            % Put the TCR grid which is dynamic, into a very large static grid
            Lgrid_lon = -100:lon_inc:-50; 
            Lgrid_lat = 0:lat_inc:60; 
            % convention is longitude (xvec) along the rows from most negative to least
            % negative and latitude (yvec) along the columns from smallest to largest
            [lYn, lXn] = meshgrid(Lgrid_lat, Lgrid_lon);
            Lgrid = zeros(length(Lgrid_lon), length(Lgrid_lat));
            for j = tstart:trk 
                % % Plot the TCR rainfall grid (dynamic grid)
                % if mod(j,5) ==0
                %     pcolor(PLONG_SAVE(:,:,j), PLAT_SAVE(:,:,j), rain(:,:,j))
                %     shading flat
                %     colorbar()
                %     hold on;
                %     plot(xvec, yvec(1), 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                %     hold on;
                %     plot(xvec, yvec(end), 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                %     hold on;
                %     plot(xvec(1), yvec, 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                %     hold on;
                %     plot(xvec(end), yvec, 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                %     stormfile = strcat(num2str(j), '_localGrid_inTCRgrid.png');
                %     saveas(gcf,sprintf('%s/%s',outputfolder,stormfile))
                %     close(gcf)
                % end
        
                % Get TCR grid info at timestep j
                tcr_lat = PLAT_SAVE(1,:,j);
                tcr_lon = PLONG_SAVE(:,1,j);
                tcr_rain = rain(:,:,j);
        
                if sum(tcr_rain, 'all') == 0
                    break
                else
                    % Put grid into larger grid
                    % J is the column index in the larger grid
                    % I is the row index in the larger grid
                    [~, J] = min(abs(Lgrid_lon - tcr_lon(1,1)));
                    [~, I] = min(abs(Lgrid_lat- tcr_lat(1,1)));
                    
                    % Location of the smaller grid in the larger grid 
                    lat_inds = I:I+size(tcr_lat,2)-1;
                    lon_inds = J:J+size(tcr_lon,1)-1; 
                    Lgrid(lon_inds, lat_inds, j) = tcr_rain;
                    
                    % if mod(j,5) == 0
                    %     pcolor(lXn, lYn, Lgrid(:,:,j))
                    %     shading flat
                    %     colorbar()
                    %     hold on;
                    %     plot(xvec, yvec(1), 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                    %     hold on;
                    %     plot(xvec, yvec(end), 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                    %     hold on;
                    %     plot(xvec(1), yvec, 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                    %     hold on;
                    %     plot(xvec(end), yvec, 'ro', 'MarkerSize', 2, 'LineWidth', 1);
                    %     stormfile = strcat(num2str(j), '_localGrid_inTCRgrid_expand.png');
                    %     saveas(gcf,sprintf('%s/%s',outputfolder,stormfile))
                    %     close(gcf)
                    % end
                end
            end
        
            % Clip the larger grid to the study area
            % Find the indices within the bounding box
            lat_idx = find(Lgrid_lat >= min_lat & Lgrid_lat <= max_lat);
            lon_idx = find(Lgrid_lon >= min_lon & Lgrid_lon <= max_lon);
        
            clipped_data = Lgrid(lon_idx, lat_idx, :);
            clipped_latitudes = lYn(lat_idx, lat_idx);
            clipped_longitudes = lXn(lon_idx, lon_idx);
        
            % Write storm rain grid to a netcdf
            x = clipped_longitudes(:,1);
            y = clipped_latitudes(1,:);
            [dim1, dim2, dim3] = size(clipped_data);
            time = linspace(1, dim3, dim3); 
        
            % Create a new NetCDF file
            [~, name, ~] = fileparts(filename);
            outFile = fullfile(outputfolder, [name, '.nc']);
            ncid = netcdf.create(outFile, 'NETCDF4');
        
            % Define the dimensions of the data
            dimid_x = netcdf.defDim(ncid, 'x', length(x));       % Define x dimension
            dimid_y = netcdf.defDim(ncid, 'y', length(y));       % Define y dimension
            dimid_time = netcdf.defDim(ncid, 'time', length(time));  % Define time dimension
        
            % Define the variables
            varid_x = netcdf.defVar(ncid, 'x', 'double', dimid_x);
            varid_y = netcdf.defVar(ncid, 'y', 'double', dimid_y);
            varid_time = netcdf.defVar(ncid, 'time', 'double', dimid_time);
            varid_precip = netcdf.defVar(ncid, 'precip', 'double', [dimid_x, dimid_y, dimid_time]);
        
            % Add some attributes
            netcdf.putAtt(ncid, varid_x, 'units', 'degrees_east');
            netcdf.putAtt(ncid, varid_y, 'units', 'degrees_north');
            netcdf.putAtt(ncid, varid_time, 'units', 'hours');
            netcdf.putAtt(ncid, varid_precip, 'units', 'mm');
        
            % End define mode
            netcdf.endDef(ncid);
        
            % Write data to the variables
            netcdf.putVar(ncid, varid_x, x);
            netcdf.putVar(ncid, varid_y, y);
            netcdf.putVar(ncid, varid_time, time);
            netcdf.putVar(ncid, varid_precip, clipped_data);
        
            % Close the NetCDF file
            netcdf.close(ncid);
        else
            continue
        end
    end
end
