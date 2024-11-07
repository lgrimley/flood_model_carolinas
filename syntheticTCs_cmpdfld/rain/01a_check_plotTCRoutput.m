rain_folder = 'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\rain\TCR_RainOutput\';
filename = [rain_folder,'0245.mat'];
load(filename)
outdir = 'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\rain\tc_245';

trk = size(rain);
PLONG_SAVE(PLONG_SAVE>180) = PLONG_SAVE(PLONG_SAVE>180)-360;


t = 5;
xt = PLONG_SAVE(:,:,t);
yt = PLAT_SAVE(:,:,t);
rt = rain(:,:,t);




for t = 1:trk(3)
    xt = PLONG_SAVE(:,:,t);
    yt = PLAT_SAVE(:,:,t);
    rt = rain(:,:,t);

    stormfile = strcat(num2str(t), '.png');
    pcolor(xt, yt, rt)
    shading flat
    colorbar()
    saveas(gcf,sprintf('%s/%s',outdir,stormfile))
    close(gcf)
end
