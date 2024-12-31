clear
clc

% get the maximum wind speed over each county 
gcm = {'canesm_ssp585'; 'cnrm6_ssp585'; 'ecearth6_ssp585'; 'ipsl6_ssp585'; 'miroc6_ssp585'}; 
cd 'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585';

for kk = 1:length(gcm)
    trackpath = fullfile('.\tracks', strcat('UScoast6_AL_', gcm{kk}, 'cal_roEst1rmEst1_trk100'));
    load(trackpath)

    selected_tracks = readtable(fullfile('.\stormTide', strcat('stormTide_TCIDs_and_gageCounts_', gcm{kk}, 'cal.csv')));

    fig_outdir = fullfile('.\wind', strcat('figs_', gcm{kk}, 'cal'));
    nc_outdir = fullfile('.\wind', strcat('CLE15_Gridded_' ,gcm{kk}, 'cal'));

    if ~isfolder(nc_outdir)
        mkdir(nc_outdir);
    end

    % if ~isfolder(fig_outdir)
    %     mkdir(fig_outdir);
    % end

    % set study area bounds
    latmin = 31.95;
    latmax = 36.81; 
    lonmin = -83.67; 
    lonmax = -75.11;
    
    % set grid increment
    % maybe change to 1km -- for an intense storm does it make a difference
    inc = 0.017; 
    lonvec = lonmin:inc:lonmax; 
    latvec = latmin:inc:latmax; 
    [Xn, Yn] = meshgrid(lonvec, latvec);
    M = size(Yn,1);
    N = size(Xn,2);
    
    % center of study area - for searching to select storms below with 200km
    % buffer
    %clat = 35.482015; 
    %clong = -77.554830;
    %select_tcs = [];
    %not_selected_tcs = [];
    
    maxwindgrid_all = zeros(M,N,1);
    it = 1;
    for a = 1:length(selected_tracks.tc_id)
        i = selected_tracks.tc_id(a);
        if i == 0
            continue
        else
    
        % Create a new NetCDF files
        stnum = i;
        if stnum < 10
            stormfile = strcat('000',num2str(stnum),'.nc');
        elseif stnum >= 10 && stnum < 100
            stormfile = strcat('00', num2str(stnum), '.nc');
        elseif stnum >= 100 && stnum < 1000
            stormfile = strcat('0', num2str(stnum), '.nc');
        else
            stormfile = strcat(num2str(stnum), '.nc');
        end 
        f = fullfile(nc_outdir,stormfile);
    
        % If this wind file has not already been processed
        if isfile(f) == 0
            disp(i)
            tr_lat = lat100(i,:); 
            tr_lat = tr_lat(tr_lat~=0); 
            tr_lon = lon100(i,:); 
            tr_lon = tr_lon(1:length(tr_lat)); 
        
            vmax = vstore100(i,1:length(tr_lat))./1.94; % knots to m/s; maximum sustained cyclonic wind speed
            vmax = vmax./0.85; % surface to gradient increase
            pmin = pstore100(i,1:length(tr_lat)); % min central pressure (hPa)
            rmax = rmw100(i,1:length(tr_lat)); % radius of max wind (km)
            ro = ro100(i,1:length(tr_lat)); % radius of vanishing cyclonic wind (outer radius; km)
            trk = length(tr_lat); % track points
            uinc = uinc100(i,1:trk); % u-v components of background wind which is added to the cyclonic wind
            vinc = vinc100(i,1:trk);
        
            % You can comment this out if you already selected the tracks you want
            % to model, otherwise this searches for storms within a distance of the
            % center of the study area
            % -------------
            % d = []; 
            % for j = 1:trk
            %     [d(j),~,~] = m_lldist_L([tr_lon(j), clong], [tr_lat(j), clat]);
            % end
            % if min(d)>400
            %     not_selected_tcs(end+1) = i;
            %     continue
            % end
            % select_tcs(end+1) = i;
        
            % ------------- END
        
            maxwindgrid = zeros(size(Xn));
            windgrid = zeros(size(Xn));
            Vwindgrid = zeros(size(Xn));
            Uwindgrid = zeros(size(Xn));
            for j = 1:trk
                vtemp = vmax(j); % maximum sustained cyclonic wind speed at loc j
                rtemp = rmax(j); % radius of max wind (km) at track loc j
                rotemp = ro(j); % radius of vanishing cyclonic wind at loc j
        
                drfracrm = .01; %calculating VV at radii relative to rmax ensures no smoothing near rmax!
                rfracrm_min = 0;    %[-]; r=0
                rfracrm_max = rotemp/rtemp;   %[-]; r=r0
                rrfracrm = rfracrm_min:drfracrm:rfracrm_max+drfracrm;
                %rr = rrfracrm*rtemp*1000;
        
                fcor = 2*7.292e-5*sind(abs(tr_lat(j))); % coriolis parameter
        
                % Using CLE15 (Chavas et al., 2015) WIND MODEL
                % Outputs: 
                %   rr [m] - vector of radii
                %   VV [ms-1] = vector of wind speeds at rr
        
                [rr, VV, ~] = ER11E04_nondim_rmaxinput( ...
                    vtemp, ... % Vmax [ms-1] - maximum wind speed
                    rtemp*1000, ... % rmax [m] - radius of Vmax
                    fcor, ... % fcor [s-1] - Coriolis parameter
                    1, ... % Cdvary [] - 1=C_d varies following Donelan et al 2004; 0=input value
                    1.5e-3, ... % C_d [-] - drag coefficient in outer region; ignored if Cdvary = 1
                    2/1000, ... % w_cool [sm-1] - radiative-subsidence rate
                    1, ... % CkCdvary [] - 1=C_k/C_d varies following quadratic fit to Vmax from Chavas et al. 2015; 0=input value
                    1, ... % CkCd [-] - ratio of surface exchange coefficients of enthalpy and momentum in inner region
                    1, ... % eye_adj [-] - 0 = use ER11 profile in eye; 1 = empirical adjustment
                    0.15 ...% alpha_eye [-] - V/Vm in eye is reduced by factor (r/rm)^alpha_eye; ignored if eye_adj=0
                    );
                close(gcf)  % Not sure what figure is trying to be plotted by the Wind Function??
    
                % Outer radius
                if VV(end)<0
                    r0 = rr(end-1); 
                else
                    r0 = rr(end); 
                end
        
                % create a grid with "inc" spacing extending to the r15 radius
                % This means it will only take wind until the radius of 15 m/s. But
                % if you want the entire wind profile, then use 
                ind = find(VV>0,1,'last');
                %ind = find(VV>15,1,'last');
                r15 = rr(ind)/1000; 
        
                % OUTPUT
                % Xgrid = longitude coordinate grid
                % Ygrid = latitude coordinate grid 
                [Xlocal, Ylocal] = BoundingGrid( ...
                    tr_lat(j), ... % latdeg = latitude of center in degrees 
                    tr_lon(j), ... % londeg = longitude of center in degrees
                    r15, ...% rkm = radius (or half length) of grid in km 
                    inc,... % inc = grid spacing increment in degrees 
                    Xn(1,1), ... % xmin
                    Xn(1,end), ... % xmax
                    Yn(1,1), ... % ymin
                    Yn(end,1) ... % ymax
                    );
    
                % what is this doing?!
                if size(Xlocal,1)==0 | size(Xlocal,2)==2
                    Vwindgrid(:,:, j) = 0;
                    Uwindgrid(:,:, j) = 0;
                    windgrid(:,:, j) = 0;
                    disp(['No wind in local grid, set grid to zero at time ', num2str(j)])
                    continue
                end
        
                % Create array to save wind speed, and U-V components to for each
                % track timestep (j)
                Wgrid = zeros(size(Xlocal));
                Vgrid = zeros(size(Xlocal));
                Ugrid = zeros(size(Xlocal));
        
                % V_back = background wind speed; added to the cyclone wind to get total
                V_back = sqrt(uinc(j).^2 + vinc(j).^2);
                due_S = [0, -10000,0];
                track_v = [uinc(j), vinc(j), 0];
                
                % this for loop is figuring out the angle of the cyclonic and
                % background wind at the surface level, resolving the cyclonic and
                % background wind speeds into their vector (x, y) components, and
                % saving them in the wind grid 
                for ii = 1:size(Xlocal,1)
                    for jj = 1:size(Ylocal,2)
                        % M_LLDIST Spherical earth distance between points in long/lat coordinates.
                        [dist, ~, ~] = m_lldist_L([Xlocal(ii,jj), tr_lon(j)], ...
                            [Ylocal(ii,jj), tr_lat(j)]);
                        V_surf = interp1(rr,VV,dist*1000).*0.85;
                        hur_v = [Xlocal(ii,jj)-tr_lon(j), Ylocal(ii,jj)-tr_lat(j),0];
                        if hur_v(1) <0 && hur_v(2)>0
                            tr_ang = atan2d(norm(cross(due_S,hur_v)), ...
                                dot(due_S, hur_v));
                        elseif hur_v(1) >0 && hur_v(2) >0
                            tr_ang = 360 - atan2d(norm(cross(due_S,hur_v)), ...
                                dot(due_S, hur_v));
                        elseif hur_v(1) >0 && hur_v(2) <0
                            tr_ang = 360 - atan2d(norm(cross(due_S,hur_v)), ...
                                dot(due_S, hur_v));
                        else 
                            tr_ang = atan2d(norm(cross(due_S,hur_v)), ...
                                dot(due_S, hur_v));
                        end
                        w_ang_gr = tr_ang - 90;
                        r_inf = dist;
                        if r_inf > 1.2*rtemp
                            inf_ang = 25;
                        else
                            x = [0, rtemp, 1.2*rtemp, 3*rtemp];
                            y = [10, 20, 25, 25];
                            inf_ang = interp1(x, y, r_inf);
                        end
                        w_ang_surf = w_ang_gr - inf_ang;
        
                        if track_v(1) <0 && track_v(2)>0
                            track_ang = atan2d(norm(cross(track_v,due_S)), ...
                                dot(track_v, due_S));
                        elseif track_v(1) >0 && track_v(2) >0
                            track_ang = 360 - atan2d(norm(cross(due_S, track_v)), ...
                                dot(track_v, due_S));
                        elseif track_v(1) >0 && track_v(2) <0
                            track_ang = 360 - atan2d(norm(cross(due_S,track_v)), ...
                                dot(track_v, due_S));
                        else 
                            track_ang = atan2d(norm(cross(due_S,track_v)), ...
                                dot(due_S, track_v));
                        end
                        back_ang = track_ang - 20;
                        
                        % Final wind components (background and cyclonic)
                        V_toty = -1*V_surf.*cosd(w_ang_surf) + V_back.*sind(back_ang);
                        V_totx = -1*V_surf.*sind(w_ang_surf) + V_back.*cosd(back_ang);
                        Wgrid(ii,jj) = sqrt(V_totx^2+V_toty^2);
                        
                        % Wind speed u-v components
                        Vgrid(ii,jj) = V_toty;
                        Ugrid(ii,jj) = V_totx;
                    end
                end
                
                % put local wind grid into larger grid
                diff_J = abs(Xn(1,:) - Xlocal(1,1));
                [~, J] = min(diff_J); % J is the column index in the larger grid
        
                 % Calculate the difference in the x,y locations of the larger grid compared to the 
                 % upper right corner x,y of the local grid. Pull the index in the larger grid
                 % of the cell that has the smallest different (e.g., matches coords best)
                diff_I = abs(Yn(:,1)-Ylocal(1,1));
                [~, I] = min(diff_I); % I is the row index in the larger grid
        
                % Location of the smaller grid in the larger grid 
                large_grid_rows = I:I+size(Ylocal,1)-1;
                large_grid_columns = J:J+size(Xlocal,2)-1; 
        
                maxwindgrid(large_grid_rows,large_grid_columns) = max(maxwindgrid(large_grid_rows,large_grid_columns), Wgrid);
                Vgrid(isnan(Vgrid)) = 0.0;
                Ugrid(isnan(Ugrid)) = 0.0;
                Wgrid(isnan(Wgrid)) = 0.0;
                Vwindgrid(large_grid_rows,large_grid_columns, j) = Vgrid;
                Uwindgrid(large_grid_rows,large_grid_columns, j) = Ugrid;
                windgrid(large_grid_rows,large_grid_columns, j) = Wgrid;
                disp(['Final wind components calculated (background and cyclonic) at tstep: ', num2str(j)])
                % if mod(j,10) == 0
                %     % Plot of the wind on the local grid scale
                %     pcolor(Xlocal, Ylocal, Wgrid)
                %     shading flat
                %     colorbar()
                %     stormfile = strcat(num2str(j), '_localGrid.png');
                %     saveas(gcf,sprintf('%s/%s',fig_outdir,stormfile))
                %     close(gcf)
                % 
                %     % Plot of the max wind speed across the larger grid so far in
                %     % the TC track
                %     pcolor(Xn, Yn, maxwindgrid)
                %     shading flat
                %     colorbar()
                %     stormfile = strcat(num2str(j), '_maxWind_largerGrid.png');
                %     saveas(gcf,sprintf('%s/%s',fig_outdir,stormfile))
                %     close(gcf)
                % 
                %     % Plot of the local wind nested in the larger grid scale,
                %     % Used for generating SFINCS model inputs w/ U-V components
                %     pcolor(Xn, Yn, windgrid(:,:,j))
                %     shading flat
                %     colorbar()
                %     stormfile = strcat(num2str(j), '_windspeed_largerGrid.png');
                %     saveas(gcf,sprintf('%s/%s',fig_outdir,stormfile))
                %     close(gcf)
                % end
            end
            
            % Write storm rain grid to a netcdf
            x = Xn(1,:);
            y = Yn(:,1);
            [dim1, dim2, dim3] = size(windgrid);
            time = linspace(1, dim3, dim3); 
        
            % Create a new NetCDF files
            ncid = netcdf.create(f, 'NETCDF4');
        
            % Define the dimensions of the data
            dimid_x = netcdf.defDim(ncid, 'x', length(x));       % Define x dimension
            dimid_y = netcdf.defDim(ncid, 'y', length(y));       % Define y dimension
            dimid_time = netcdf.defDim(ncid, 'time', length(time));  % Define time dimension
        
            % Define the variables
            varid_x = netcdf.defVar(ncid, 'x', 'double', dimid_x);
            varid_y = netcdf.defVar(ncid, 'y', 'double', dimid_y);
            varid_time = netcdf.defVar(ncid, 'time', 'double', dimid_time);
            varid_wndspd = netcdf.defVar(ncid, 'wind_speed', 'double', [dimid_x, dimid_y, dimid_time]);
            varid_U = netcdf.defVar(ncid, 'wind10_u', 'double', [dimid_x, dimid_y, dimid_time]);
            varid_V = netcdf.defVar(ncid, 'wind10_v', 'double', [dimid_x, dimid_y, dimid_time]);
        
            % Add some attributes
            netcdf.putAtt(ncid, varid_x, 'units', 'degrees lon');
            netcdf.putAtt(ncid, varid_y, 'units', 'degrees lat');
            %netcdf.putAtt(ncid, varid_time, 'units', 'hours');
            netcdf.putAtt(ncid, varid_wndspd, 'units', 'm/s');
        
            % End define mode
            netcdf.endDef(ncid);
        
            % Write data to the variables
            netcdf.putVar(ncid, varid_x, x);
            netcdf.putVar(ncid, varid_y, y);
            netcdf.putVar(ncid, varid_time, time);
            netcdf.putVar(ncid, varid_wndspd, permute(windgrid, [2,1,3]));
            netcdf.putVar(ncid, varid_U, permute(Uwindgrid, [2,1,3]));
            netcdf.putVar(ncid, varid_V, permute(Vwindgrid, [2,1,3]));
        
            % Close the NetCDF file
            netcdf.close(ncid);
        
            % Save the max wind for this TC to master array
            maxwindgrid_all(:,:,it) = maxwindgrid;
            it = it + 1;
            disp(['Done processing TC: ', num2str(i)])
        else
            continue
        end
        end
    end
end
