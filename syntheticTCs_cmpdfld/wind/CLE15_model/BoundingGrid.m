function [Xgrid, Ygrid] = BoundingGrid(latdeg, londeg, rkm, inc, xmin, ...
    xmax, ymin, ymax)
% INPUT 
% latdeg = latitude of center in degrees 
% londeg = longitude of center in degrees
% rkm = radius (or half length) of grid in km 
% inc = grid spacing increment in degrees 

% OUTPUT
% Xgrid = longitude coordinate grid
% Ygrid = latitude coordinate grid 

% Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0;  % Major semiaxis [m]
WGS84_b = 6356752.3;  % Minor semiaxis [m]

% convert degree to radians
lat = deg2rad(latdeg);
lon = deg2rad(londeg);

% Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
An = WGS84_a*WGS84_a * cos(lat);
Bn = WGS84_b*WGS84_b * sin(lat);
Ad = WGS84_a * cos(lat);
Bd = WGS84_b * sin(lat);
radius = sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) );
pradius = radius*cos(lat);

% create bounding box in degrees 
halfside = rkm*1000; 
latMin = round(rad2deg(lat - halfside/radius)*(1/inc))/(1/inc);
if ymin ~=0 
    latMin = max(latMin,ymin);
end
latMax = round(rad2deg(lat + halfside/radius)*(1/inc))/(1/inc);
if ymax ~=0 
    latMax = min(latMax,ymax);
end
lonMin = round(rad2deg(lon - halfside/pradius)*(1/inc))/(1/inc);
if xmin ~=0 
    lonMin = max(lonMin,xmin);
end
lonMax = round(rad2deg(lon + halfside/pradius)*(1/inc))/(1/inc);
if xmax ~=0 
    lonMax = min(lonMax,xmax);
end

% create grid with specified spacing (inc)
Xvec = lonMin:inc:lonMax; 
Yvec = latMin:inc:latMax;

[Xgrid, Ygrid] = meshgrid(Xvec, Yvec);
end

