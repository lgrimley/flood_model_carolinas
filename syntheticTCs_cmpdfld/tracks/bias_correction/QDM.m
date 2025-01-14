% quantile delta matching 

function [xphat, xhhat, xphat2] = QDM(xoh, xmh, xmp, h)
% historical observed distribution 
[fit_oh, ~, ~] = fit_gpd(xoh, round(length(xoh)*0.01), h);
%[fit_oh, ~, ~] = fit_gpd(xoh, 8);
% historical modeled distribution 
[fit_mh, ~, ~] = fit_gpd(xmh, round(length(xmh)*0.01), h);
%[fit_mh, ~, ~] = fit_gpd(xmh, 8);
% projected modeled distribution 
[fit_mp, ~, ~] = fit_gpd(xmp, round(length(xmp)*0.01), h);
%[fit_mp, ~, ~] = fit_gpd(xmp, 8);
xhhat = icdf(fit_oh, cdf(fit_mh, xmh)); 

% relative change between historical and future 
delt = xmp./icdf(fit_mh, cdf(fit_mp, xmp)); 
delt2 = xmp - icdf(fit_mh, cdf(fit_mp, xmp)); 
delt2(delt2<-10^10) = xmp(delt2<-10^10); 

% future target distribution 
%xphat = icdf(fit_oh, cdf(fit_mp, xmp)).*delt; 
xphat = icdf(fit_oh, cdf(fit_mp, xmp)) + delt2; 
xphat(xphat<0) = xmp(xphat<0);
end

