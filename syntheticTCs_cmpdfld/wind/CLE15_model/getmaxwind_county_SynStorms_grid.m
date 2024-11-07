clear all
clc

% get the maximum wind speed over each county 
cd 'Z:\Data-Expansion\users\lelise\Chapter3\tracks\NCEP'
load UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100

% add directory where wind functions/scripts stored
cd 'Z:\Data-Expansion\users\lelise\Chapter3\wind\scripts'

% set study area bounds -- currently set for entire east coast
latmin = 25.7;
latmax = 37.5; 
lonmin = -85.86; 
lonmax = -69.49;

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
clat = 35.482015; 
clong = -77.554830;

maxwindgrid_all = zeros(M,N,1);
it = 1;
select_tcs = [];
not_selected_tcs = [];
for i = 1:length(year100(:,1))
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
    d = []; 
    for j = 1:trk
        [d(j),~,~] = m_lldist_L([tr_lon(j), clong], [tr_lat(j), clat]);
    end
    if min(d)>400
        not_selected_tcs(end+1) = i;
        continue
    end
    select_tcs(end+1) = i;

    % ------------- END

    maxwindgrid = zeros(size(Xn));
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
        close(gcf)
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
        if size(Xlocal,1)==0 | size(Xlocal,2)==2
            continue
        end
        Wgrid = zeros(size(Xlocal));

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
            end
        end
        % Plot
        %pcolor(Xn, Yn, Wgrid)
        %shading flat
        %saveas(gcf,'Z:\Data-Expansion\users\lelise\Chapter3\rain\wind_158_tstep75.png')
        %close(gcf)

        % put local wind grid into larger grid
        
% edit code to find the minimum distance between x,y then use that as the
% J, I index

        J = find(round(Xn,1)==round(Xlocal(1,1),1));
        I = find(Yn == Ylocal(1,1));
        maxwindgrid(I:I+size(Ylocal,1)-1,J:J+size(Xlocal,2)-1) = ...
            max(maxwindgrid(I:I+size(Ylocal,1)-1,J:J+size(Xlocal,2)-1), ...
            Wgrid);

    end
    %pcolor(Xn, Yn, maxwindgrid)
    %shading flat

    maxwindgrid_all(:,:,it) = maxwindgrid; 
    it = it + 1;
    disp(it)
end
fileout = 'Z:\Data-Expansion\users\lelise\Chapter3\wind\NCEP_TCselect_400km_buffer.csv';
writematrix(select_tcs, fileout)
%save('maxwindmat_ncep.mat', 'maxwindgrid_all')        