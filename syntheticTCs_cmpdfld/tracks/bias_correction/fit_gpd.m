% function for selecting threshold and fitting gpd distribution 

function [fit, u, type] = fit_gpd(data, minp, h)
data = data(data~=Inf);
data = sort(data);
n = length(data);
myfun1 = @(x)ksdensity(x,'Bandwidth',h,'Function','cdf');
if 0.2*n<minp
    fit = fitdist(data', 'Kernel', 'Bandwidth', h); 
    u = 0; 
    type = 'kde'; 
else
    upvec = round(0.8*n):1:n+1-minp;
    pvec = upvec/(n+1);
    obs_cdf = 1/(n+1):1/(n+1):n/(n+1);
    err = zeros(length(upvec),1); 
    for i = 1:length(upvec)
        warning('')
        pfit = paretotails(data, 0,pvec(i), 'kernel');
        [warnMsg, warnId] = lastwarn;
        if sum(isnan(pfit.UpperParameters))>0
            err(i) = Inf;
        elseif ~isempty(warnMsg)
            err(i) = Inf;
        else
            mod_cdf = cdf(pfit, data); 
            mod_cdf = mod_cdf(data>=data(upvec(i)));
            obs = obs_cdf(data>=data(upvec(i)));
            nn = length(obs);
            err(i) = sqrt(sum((mod_cdf-obs').^2)/nn);
        end
    end
    if min(err) == Inf
        fit = fitdist(data', 'Kernel', 'Bandwidth', h);
        u = 0;
        type = 'kde';
    else
        u = pvec(err==min(err));
        fit = paretotails(data, 0, u, myfun1); 
        err = min(err);
        type = 'gp';
    end
end

