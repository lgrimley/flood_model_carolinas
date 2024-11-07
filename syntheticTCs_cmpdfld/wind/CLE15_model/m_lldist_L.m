function [dist, dx, dy] = m_lldist_L(long,lat)
% long in negtive coordinate
% M_LLDIST Spherical earth distance between points in long/lat coordinates. 
%   RANGE=M_LLDIST(LONG,LAT) gives the distance in kilometers between
%   successive points in the vectors LONG and LAT, computed
%   using the Haversine formula on a spherical earth of radius
%   6378.137km. Distances are probably good to better than 1% of the
%   "true" distance on the ellipsoidal earth

%   http://www.movable-type.co.uk/scripts/latlong.html
%

pifac=pi/180;
earth_radius=6378.137;

long1=long(1:end-1)*pifac;
long2=long(2:end)*pifac;
lat1= lat(1:end-1)*pifac;
lat2= lat(2:end)*pifac;

dlon = long2 - long1; 
dlat = lat2 - lat1; 

% ?haversine? formula

% a = (sin(dlat/2)).^2 + cos(lat1) .* cos(lat2) .* (sin(dlon/2)).^2;
% angles = 2 * atan2( sqrt(a), sqrt(1-a) );
% dist = earth_radius * angles;

dist=earth_radius*2*asin(sqrt(sin(dlat/2).^2 + cos(lat1).*cos(lat2).*(sin(dlon/2)).^2));

% dx 
dx=earth_radius*2*asin(sqrt(cos(lat1).*cos(lat2).*(sin(dlon/2)).^2)).*sign(dlon);
dy=earth_radius*2*asin(sqrt(sin(dlat/2).^2)).*sign(dlat); % get correct sign

% note the longitude coordinate for the right sign of dx dy
