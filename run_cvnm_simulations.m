function outDir = run_cvnm_simulations(mode,outDir)
%RUN_CVNM_SIMULATIONS  Bit error probability of balanced variance codes.
%
%   out = run_cvnm_simulations;          % 20 million bit trials per point
%   out = run_cvnm_simulations('smoke'); % short run to check the setup
%   run_cvnm_simulations('redraw',out);  % reuse the numerical results in out
%
% Fig. 2 compares the detectors in AWGN and flat Rayleigh fading, then tests
% DC offset and I/Q imbalance. Fig. 3 examines what changes when the gain
% varies within a bit and the variance patterns are no longer interchangeable.
% Each run saves its error counts and figures under cvnm_results/.
if nargin<1 || isempty(mode), mode='full'; end
mode=validatestring(mode,{'full','smoke','redraw'});
if strcmp(mode,'redraw')
    if nargin<2 || ~isfile(fullfile(outDir,'simulation_complete.mat'))
        error('CVNM:Redraw','Supply the completed output folder from run_cvnm_simulations.');
    end
    prior=load(fullfile(outDir,'simulation_complete.mat'),'preset');
    single_plot(outDir,prior.preset);
    fprintf('Redrew figures in %s. No Monte Carlo trials were run.\n',outDir);
    return
end
if nargin<2 || isempty(outDir)
    outDir=fullfile(pwd,'cvnm_results',[mode '_' datestr(now,'yyyymmdd_HHMMSSFFF')]);
end
outDir=char(outDir);
if isfolder(outDir) || isfile(outDir)
    error('CVNM:OutputExists','Choose a new output folder: %s',outDir);
end
[ok,msg]=mkdir(outDir); assert(ok,'%s',msg);
preset=mode; nBits=20000000; if strcmp(mode,'smoke'), nBits=2000; end
previousRng=rng; keepRng=onCleanup(@()rng(previousRng)); %#ok<NASGU>
clock=tic;
nominal=fullfile(outDir,'nominal');
front=fullfile(outDir,'frontend'); [ok,msg]=mkdir(front); assert(ok,'%s',msg);
order=fullfile(outDir,'order');
fprintf('CVNM / %s: %s trials per scheme and setting. Output: %s\n',preset,num2str(nBits),outDir);
run_cvnm(preset,'all',nominal);     % Figs. 2(a,b), 3(a,b)
single_frontend(preset,front);     % Fig. 2(c) and AWGN low-BEP checkpoints
run_fig3c(preset,order);           % Fig. 3(c)
summarize_fig3(nominal,nominal,order,fullfile(outDir,'summary'),nBits);
save(fullfile(outDir,'simulation_complete.mat'),'preset','nBits');
single_plot(outDir,preset);
fprintf('Finished in %.1f minutes. Figures: %s\n',toc(clock)/60,outDir);
end

function y=sb_gamma(shape,nRows,nCols)
% Unit-scale Gamma via Marsaglia and Tsang. All shapes used here are >= 1.
% Only replace rejected elements, keeping memory bounded by the batch size.
assert(isscalar(shape) && shape>=1 && isfinite(shape),'Invalid Gamma shape.');
d=shape-1/3; c=1/sqrt(9*d);
y=zeros(nRows,nCols); left=(1:numel(y)).';
while ~isempty(left)
    z=randn(numel(left),1); base=1+c*z; v=base.^3;
    u=rand(numel(left),1); take=false(size(left));
    valid=base>0;
    take(valid)=log(max(u(valid),realmin)) < ...
        0.5*z(valid).^2+d*(1-v(valid)+log(v(valid)));
    y(left(take))=d*v(take);
    left=left(~take);
end
end

function outDir = run_cvnm(preset, selection, outDir)
% Figs. 2(a,b) and 3(a,b): predictions and error counts for each receiver.
if nargin<1 || isempty(preset), preset = 'full'; end
if nargin<2 || isempty(selection), selection = 'all'; end
preset = validatestring(preset, {'smoke','reference','full'});
selection = validatestring(selection, {'all','2ab-theory','2ab-mc','3a','3b'});
if nargin<3 || isempty(outDir)
    outDir = fullfile(pwd,'runs',[preset '_' selection '_' datestr(now,'yyyymmdd_HHMMSSFFF')]);
end
if isfolder(outDir) || isfile(outDir)
    error('CVNM:OutputExists','Choose a new output directory: %s',char(outDir));
end
[ok,msg] = mkdir(outDir); assert(ok,'%s',msg);
initialRandomState = rng;
restoreRandomState = onCleanup(@() rng(initialRandomState)); %#ok<NASGU>

% Normalize received energies by N0. At h=1, the noise variance is then 1
% and the average signal variance per sample is (Eb/N0)/N.
cfg.seed = 20260802;
cfg.N = 64; cfg.M = cfg.N/2; cfg.delta = 1;
cfg.noiseModAlpha = 10; cfg.deltaNc = (cfg.noiseModAlpha-1)/(cfg.noiseModAlpha+1);
% Spread the energy of L data bits across L+1 transmitted DBN intervals:
% the extra interval provides the first reference and is not a free symbol.
cfg.dbnFrameBits = 100; cfg.dbnIntervalEnergy = cfg.dbnFrameBits/(cfg.dbnFrameBits+1);
cfg.rhoBMain = 0.5; cfg.timeFixedDb = 18;
cfg.awgnDbMc = [0 2 4 6 8 9 10 11 12 13 14 14.5 15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 20];
cfg.rayleighDbMc = 0:2.5:30;
cfg.rhoBGrid = [0.02 0.035 0.05 0.075 0.10 0.15 0.20 0.275 0.35 0.425 ...
    0.50 0.575 0.65 0.725 0.80 0.85 0.90 0.935 0.97 0.985 1];
cfg.theoryDb = 0:0.1:30;
cfg.timeDbTheory = 0:0.5:20;
cfg.timeDbMc = [0 2.5 5 7.5 10 12.5 15 16.5 18 19 20];
cfg.awgnTrials = 2e7; cfg.rayleighTrials = 2e7;
cfg.timeTrials = 2e7; cfg.timeTheoryTrials = 2e4;
if strcmp(preset,'reference')
    % Historical trial budget for the time-selective sweeps only.
    cfg.timeTrials = 400000;
elseif strcmp(preset,'smoke')
    cfg.rhoBGrid = [0.5 1]; cfg.awgnDbMc = [0 18]; cfg.rayleighDbMc = [0 18];
    cfg.theoryDb = [0 18 30]; cfg.timeDbTheory = [0 18]; cfg.timeDbMc = [0 18];
    cfg.awgnTrials = 2000; cfg.rayleighTrials = 2000;
    cfg.timeTrials = 2000; cfg.timeTheoryTrials = 80;
end
cfg.preset = preset; cfg.selection = selection; cfg.Eb = 1;
cfg.matlabVersion = version; cfg.generator = 'twister';
fid = fopen(fullfile(outDir,'settings.json'),'wt'); assert(fid>=0,'Cannot write simulation settings.');
cleanupFile = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(cfg)); clear cleanupFile
fprintf('Preset %s / %s. Writing results to %s\n',preset,selection,char(outDir));

if any(strcmp(selection,{'all','2ab-theory'}))
    fprintf('Computing exact AWGN and flat-Rayleigh benchmarks ...\n');
    nominal_theory_table(cfg,outDir);
end
if any(strcmp(selection,{'all','2ab-mc'}))
    fprintf('Sampling standard AWGN and flat-Rayleigh markers ...\n');
    rng(cfg.seed,'twister');
    [a,b,c,d,e] = mc_awgn(cfg.awgnDbMc,cfg);
    tableAWGN = table(cfg.awgnDbMc(:),a(:),b(:),c(:),d(:),e(:), ...
        'VariableNames',{'EbN0_dB','CVNM','NC_NoiseMod','OODN','NoiseMod','DBN'});
    write_mc(tableAWGN,outDir,'awgn',cfg.awgnTrials);
    [a,b,c,d,e,f] = mc_rayleigh(cfg.rayleighDbMc,cfg);
    tableRay = table(cfg.rayleighDbMc(:),a(:),b(:),c(:),d(:),e(:),f(:), ...
        'VariableNames',{'EbN0_dB','CVNM','NC_NoiseMod','DBN','OODN_perfect_CSI','OODN_fixed','NoiseMod_fixed'});
    write_mc(tableRay,outDir,'ray',cfg.rayleighTrials);
end
if any(strcmp(selection,{'all','3a'}))
    fprintf('Fig. 3(a): computing the SNR sweep ...\n');
    % Fit the fixed energy thresholds on the channel bank used for theory;
    % estimate their error rates on independent channel and noise draws.
    t = time_selective_all_theory(cfg.rhoBMain,cfg.timeDbTheory,cfg,cfg.seed+2);
    m = time_selective_all_theory(cfg.rhoBMain,cfg.timeDbMc,cfg,cfg.seed+2);
    v = mc_time_selective_all(cfg.rhoBMain,cfg.timeDbMc,m,cfg,cfg.seed+4);
    write_time(t,v,m,cfg.timeDbTheory,cfg.timeDbMc,'EbN0_dB',outDir,'time_snr',cfg.timeTrials);
end
if any(strcmp(selection,{'all','3b'}))
    fprintf('Fig. 3(b): computing the interblock-correlation sweep ...\n');
    % At each correlation, use separate channel draws for the curve, the
    % threshold fitted for the simulated receiver, and the error count.
    t = time_selective_all_theory(cfg.rhoBGrid,cfg.timeFixedDb,cfg,cfg.seed+6);
    m = time_selective_all_theory(cfg.rhoBGrid,cfg.timeFixedDb,cfg,cfg.seed+7);
    v = mc_time_selective_all(cfg.rhoBGrid,cfg.timeFixedDb,m,cfg,cfg.seed+8);
    write_time(t,v,m,cfg.rhoBGrid,cfg.rhoBGrid,'rho_b',outDir,'time_rho',cfg.timeTrials);
end
fprintf('Done. Numerical tables: %s\n',char(outDir));
end

function nominal_theory_table(cfg,outDir)
g = 10.^(cfg.theoryDb/10); N=cfg.N; alpha=cfg.noiseModAlpha; E=cfg.dbnIntervalEnergy;
% OODN has two Rayleigh benchmarks: a threshold chosen for each gain (CSI)
% and one threshold per average Eb/N0, minimizing the fading-averaged BEP.
y = zeros(numel(g),15); y(:,1) = cfg.theoryDb(:);
y(:,2) = reshape(cvnm_awgn_exact(g,N,cfg.delta),[],1);
y(:,3) = reshape(cvnm_awgn_gaussian(g,N,cfg.delta),[],1);
y(:,4) = reshape(cvnm_awgn_exact(g,N,cfg.deltaNc),[],1);
y(:,5) = reshape(oodn_awgn_exact(g,N),[],1);
y(:,6) = reshape(noisemod_awgn_exact(g,N,alpha),[],1);
y(:,7) = reshape(dbn_awgn_exact(g,N,E),[],1);
for k=1:numel(g)
    y(k,8) = cvnm_rayleigh_exact(g(k),N,cfg.delta);
    y(k,9) = cvnm_rayleigh_exact(g(k),N,cfg.deltaNc);
    y(k,10) = dbn_rayleigh_exact(g(k),N,E);
    y(k,11) = oodn_rayleigh_csi(g(k),N);
    [y(k,12),y(k,14)] = oodn_rayleigh_fixed(g(k),N);
    [y(k,13),y(k,15)] = noisemod_rayleigh_fixed(g(k),N,alpha);
end
names={'EbN0_dB','CVNM_AWGN_exact','CVNM_AWGN_Gaussian','NC_NoiseMod_AWGN_exact', ...
    'OODN_AWGN_exact','NoiseMod_AWGN_exact','DBN_AWGN_exact','CVNM_Rayleigh_exact', ...
    'NC_NoiseMod_Rayleigh_exact','DBN_Rayleigh_exact','OODN_Rayleigh_perfect_CSI', ...
    'OODN_Rayleigh_fixed','NoiseMod_Rayleigh_fixed','OODN_fixed_threshold','NoiseMod_fixed_threshold'};
writetable(array2table(y,'VariableNames',names),fullfile(outDir,'nominal_theory.csv'));
end

function write_time(t,v,m,ax1,ax2,field,outDir,stem,n)
names = {field,'I_CVNM','contiguous_CVNM','NC_NoiseMod','OODN_statistical_threshold','NoiseMod_alpha10_statistical_threshold','DBN_L100'};
namesExtra = [names,{'OODN_threshold','NoiseMod_threshold'}];
vals = @(s) [s.I_CVNM(:),s.contiguous(:),s.NC_NoiseMod(:),s.OODN(:),s.NoiseMod(:),s.DBN(:)];
tt = array2table([ax1(:),vals(t),t.OODN_tau(:),t.NoiseMod_tau(:)],'VariableNames',namesExtra);
writetable(tt,fullfile(outDir,[stem '_theory.csv']));
fitTable = array2table([ax2(:),m.OODN_tau(:),m.NoiseMod_tau(:)], ...
    'VariableNames',{field,'OODN_threshold','NoiseMod_threshold'});
writetable(fitTable,fullfile(outDir,[stem '_thresholds_used_by_mc.csv']));
tm = array2table([ax2(:),vals(v)],'VariableNames',names);
write_mc(tm,outDir,stem,n);
end

function write_mc(t,outDir,stem,n)
writetable(t,fullfile(outDir,[stem '_mc.csv']));
counts = t; counts{:,2:end} = round(t{:,2:end}*n);
counts.BitsPerScheme = repmat(n,height(t),1);
writetable(counts,fullfile(outDir,[stem '_error_counts.csv']));
end

function pe = cvnm_awgn_exact(gamma,N,delta)
% The two group energies have Gamma shape M. After dividing by their
% respective scales, the fraction in the first group is Beta(M,M).
M = N/2;
z = (1 + gamma.*(1-delta)/N) ./ (2*(1 + gamma/N));
pe = betainc(z,M,M);
end

function pe = cvnm_awgn_gaussian(gamma,N,delta)
M = N/2;
vH = 1 + gamma.*(1+delta)/N;
vL = 1 + gamma.*(1-delta)/N;
argument = sqrt(M).*(vH-vL)./sqrt(vH.^2+vL.^2);
pe = 0.5*erfc(argument/sqrt(2));
end

function pe = oodn_awgn_exact(gamma,N)
v1 = 1 + 2*gamma/N;
tau = N.*log(v1)./(1-1./v1);
pe = 0.5*(gammainc(tau,N,'upper') + gammainc(tau./v1,N,'lower'));
end

function pe = noisemod_awgn_exact(gamma,N,alpha)
v0 = 1 + 2*gamma/(N*(1+alpha));
v1 = 1 + 2*alpha*gamma/(N*(1+alpha));
tau = N.*v0.*v1./(v1-v0).*log(v1./v0);
pe = 0.5*(gammainc(tau./v0,N,'upper') + ...
    gammainc(tau./v1,N,'lower'));
end

function pe = dbn_awgn_exact(gamma,N,intervalEnergy)
% intervalEnergy is Es/Eb, including the cost of the first reference.
z = 1./(2*(1+intervalEnergy*gamma/N));
pe = betainc(z,N,N);
end

function pe = cvnm_rayleigh_exact(gamma,N,delta)
% The instantaneous bit SNR t = gamma*|h|^2 is exponential with mean gamma.
integrand = @(t) cvnm_awgn_exact(t,N,delta).*exp(-t/gamma)/gamma;
pe = integral(integrand,0,Inf,'AbsTol',2e-11,'RelTol',2e-9);
end

function pe = dbn_rayleigh_exact(gamma,N,intervalEnergy)
integrand = @(t) dbn_awgn_exact(t,N,intervalEnergy).*exp(-t/gamma)/gamma;
pe = integral(integrand,0,Inf,'AbsTol',2e-11,'RelTol',2e-9);
end

function pe = oodn_rayleigh_csi(gamma,N)
integrand = @(t) oodn_conditional_from_t(t,N).*exp(-t/gamma)/gamma;
pe = integral(integrand,0,Inf,'AbsTol',2e-11,'RelTol',2e-9);
end

function pe = oodn_conditional_from_t(t,N)
pe = 0.5*ones(size(t));
idx = t > 1e-12;
v1 = 1 + 2*t(idx)/N;
tau = N.*log(v1)./(1-1./v1);
pe(idx) = 0.5*(gammainc(tau,N,'upper') + ...
    gammainc(tau./v1,N,'lower'));
end

function [pe,tau] = oodn_rayleigh_fixed(gamma,N)
objective = @(threshold) fixed_threshold_bep(threshold,gamma,N);
options = optimset('TolX',1e-7,'Display','off');
[tau,pe] = fminbnd(objective,0.5*N,N+4*gamma,options);
end

function pe = fixed_threshold_bep(tau,gamma,N)
% The silent OODN bit contains noise alone; only the miss probability
% needs an average over the channel gain.
pFalseAlarm = gammainc(tau,N,'upper');
integrand = @(t) gammainc(tau./(1+2*t/N),N,'lower') .* ...
    exp(-t/gamma)/gamma;
pMiss = integral(integrand,0,Inf,'AbsTol',2e-9,'RelTol',2e-8);
pe = 0.5*(pFalseAlarm+pMiss);
end

function [pe,tau] = noisemod_rayleigh_fixed(gamma,N,alpha)
objective = @(threshold) noisemod_fixed_threshold_bep( ...
    threshold,gamma,N,alpha);
options = optimset('TolX',1e-7,'Display','off');
upper = N+2*alpha*gamma/(1+alpha);
[tau,pe] = fminbnd(objective,0.5*N,upper,options);
end

function pe = noisemod_fixed_threshold_bep(tau,gamma,N,alpha)
integrand = @(t) noisemod_conditional_threshold(t,tau,N,alpha).* ...
    exp(-t/gamma)/gamma;
pe = integral(integrand,0,Inf,'AbsTol',2e-9,'RelTol',2e-8);
end

function pe = noisemod_conditional_threshold(t,tau,N,alpha)
v0 = 1+2*t/(N*(1+alpha));
v1 = 1+2*alpha*t/(N*(1+alpha));
pe = 0.5*(gammainc(tau./v0,N,'upper') + ...
    gammainc(tau./v1,N,'lower'));
end

function [cvnm,ncNoiseMod,oodn,noiseMod,dbn] = mc_awgn(dbPoints,cfg)
% Sample group energies directly: summing M independent exponential
% energies of scale v has exactly the same law as v*Gamma(M,1).
% For balanced codes the error probability is the same for either bit,
% so each trial only checks whether the high-energy group loses.
cvnm = zeros(size(dbPoints));
ncNoiseMod = zeros(size(dbPoints));
oodn = zeros(size(dbPoints));
noiseMod = zeros(size(dbPoints));
dbn = zeros(size(dbPoints));
batch = 2e5;
for k = 1:numel(dbPoints)
    gamma = 10^(dbPoints(k)/10);
    vHighCvnm = 1 + gamma*(1+cfg.delta)/cfg.N;
    vLowCvnm = 1 + gamma*(1-cfg.delta)/cfg.N;
    vHighNc = 1 + gamma*(1+cfg.deltaNc)/cfg.N;
    vLowNc = 1 + gamma*(1-cfg.deltaNc)/cfg.N;
    vOodn = 1 + 2*gamma/cfg.N;
    tau = cfg.N*log(vOodn)/(1-1/vOodn);
    v0NoiseMod = 1+2*gamma/(cfg.N*(1+cfg.noiseModAlpha));
    v1NoiseMod = 1+2*cfg.noiseModAlpha*gamma/ ...
        (cfg.N*(1+cfg.noiseModAlpha));
    tauNoiseMod = cfg.N*v0NoiseMod*v1NoiseMod/ ...
        (v1NoiseMod-v0NoiseMod)*log(v1NoiseMod/v0NoiseMod);
    vDbn = 1+2*cfg.dbnIntervalEnergy*gamma/cfg.N;
    errCvnm = 0;
    errNc = 0;
    errOodn = 0;
    errNoiseMod = 0;
    errDbn = 0;
    done = 0;
    while done < cfg.awgnTrials
        q = min(batch,cfg.awgnTrials-done);
        high = sb_gamma(cfg.M,q,1)*vHighCvnm;
        low = sb_gamma(cfg.M,q,1)*vLowCvnm;
        errCvnm = errCvnm + sum(high<low);
        highNc = sb_gamma(cfg.M,q,1)*vHighNc;
        lowNc = sb_gamma(cfg.M,q,1)*vLowNc;
        errNc = errNc + sum(highNc<lowNc);
        bits = rand(q,1)>=0.5;
        scale = 1 + double(bits)*(vOodn-1);
        energy = sb_gamma(cfg.N,q,1).*scale;
        errOodn = errOodn + sum((energy>=tau)~=bits);
        bitsNoiseMod = rand(q,1)>=0.5;
        scaleNoiseMod = v0NoiseMod + ...
            double(bitsNoiseMod)*(v1NoiseMod-v0NoiseMod);
        energyNoiseMod = sb_gamma(cfg.N,q,1).*scaleNoiseMod;
        errNoiseMod = errNoiseMod + ...
            sum((energyNoiseMod>=tauNoiseMod)~=bitsNoiseMod);
        % For DBN the sum and difference of aligned reference/data samples
        % give independent energy groups; only the sum contains the signal.
        errDbn = errDbn + sum(sb_gamma(cfg.N,q,1)*vDbn < ...
            sb_gamma(cfg.N,q,1));
        done = done+q;
    end
    cvnm(k) = errCvnm/done;
    ncNoiseMod(k) = errNc/done;
    oodn(k) = errOodn/done;
    noiseMod(k) = errNoiseMod/done;
    dbn(k) = errDbn/done;
end
end

function [cvnm,ncNoiseMod,dbn,oodnCsi,oodnFixed,noiseModFixed] = ...
    mc_rayleigh(dbPoints,cfg)
cvnm = zeros(size(dbPoints));
ncNoiseMod = zeros(size(dbPoints));
dbn = zeros(size(dbPoints));
oodnCsi = zeros(size(dbPoints));
oodnFixed = zeros(size(dbPoints));
noiseModFixed = zeros(size(dbPoints));
batch = 1e5;
for k = 1:numel(dbPoints)
    gamma = 10^(dbPoints(k)/10);
    [~,tauOodnFixed] = oodn_rayleigh_fixed(gamma,cfg.N);
    [~,tauNoiseModFixed] = noisemod_rayleigh_fixed( ...
        gamma,cfg.N,cfg.noiseModAlpha);
    errors = 0; errorsNc = 0; errorsDbn = 0; errorsOodnCsi = 0;
    errorsOodnFixed = 0; errorsNoiseModFixed = 0;
    done = 0;
    while done < cfg.rayleighTrials
        q = min(batch,cfg.rayleighTrials-done);
        % One gain per trial, constant over the bit. The pair-coherent DBN
        % benchmark uses that same fade over both adjacent intervals.
        g = -log(max(rand(q,1),realmin));
        high = sb_gamma(cfg.M,q,1).*(1+ ...
            gamma*g*(1+cfg.delta)/cfg.N);
        low = sb_gamma(cfg.M,q,1).*(1+ ...
            gamma*g*(1-cfg.delta)/cfg.N);
        errors = errors+sum(high<low);
        highNc = sb_gamma(cfg.M,q,1).*(1+ ...
            gamma*g*(1+cfg.deltaNc)/cfg.N);
        lowNc = sb_gamma(cfg.M,q,1).*(1+ ...
            gamma*g*(1-cfg.deltaNc)/cfg.N);
        errorsNc = errorsNc+sum(highNc<lowNc);
        highDbn = sb_gamma(cfg.N,q,1).*(1+ ...
            2*cfg.dbnIntervalEnergy*gamma*g/cfg.N);
        lowDbn = sb_gamma(cfg.N,q,1);
        errorsDbn = errorsDbn+sum(highDbn<lowDbn);

        bitsOodn = rand(q,1)>=0.5;
        v1Oodn = 1+2*gamma*g/cfg.N;
        energyOodn = sb_gamma(cfg.N,q,1).* ...
            (1+double(bitsOodn).*(v1Oodn-1));
        % Apply both thresholds to the same OODN energies and bits. Only
        % the CSI benchmark adapts its threshold to this trial's gain.
        tauOodnCsi = cfg.N*ones(q,1);
        idx = v1Oodn > 1+1e-12;
        tauOodnCsi(idx) = cfg.N*log(v1Oodn(idx))./ ...
            (1-1./v1Oodn(idx));
        errorsOodnCsi = errorsOodnCsi+sum( ...
            (energyOodn>=tauOodnCsi)~=bitsOodn);
        errorsOodnFixed = errorsOodnFixed+sum( ...
            (energyOodn>=tauOodnFixed)~=bitsOodn);

        bitsNoiseMod = rand(q,1)>=0.5;
        v0NoiseMod = 1+2*gamma*g/ ...
            (cfg.N*(1+cfg.noiseModAlpha));
        v1NoiseMod = 1+2*cfg.noiseModAlpha*gamma*g/ ...
            (cfg.N*(1+cfg.noiseModAlpha));
        energyNoiseMod = sb_gamma(cfg.N,q,1).* ...
            (v0NoiseMod+double(bitsNoiseMod).*(v1NoiseMod-v0NoiseMod));
        errorsNoiseModFixed = errorsNoiseModFixed+sum( ...
            (energyNoiseMod>=tauNoiseModFixed)~=bitsNoiseMod);
        done = done+q;
    end
    cvnm(k) = errors/done;
    ncNoiseMod(k) = errorsNc/done;
    dbn(k) = errorsDbn/done;
    oodnCsi(k) = errorsOodnCsi/done;
    oodnFixed(k) = errorsOodnFixed/done;
    noiseModFixed(k) = errorsNoiseModFixed/done;
end
end

function time = time_selective_all_theory(rhoBValues,dbPoints,cfg,seed)
% At each bit-spaced correlation, use the same fades for all schemes and SNRs.
rng(seed,'twister');
rhoBValues = rhoBValues(:).';
dbPoints = dbPoints(:).';
fields = {'I_CVNM','contiguous','NC_NoiseMod','OODN','NoiseMod','DBN', ...
    'OODN_tau','NoiseMod_tau'};
for f = 1:numel(fields)
    time.(fields{f}) = zeros(numel(rhoBValues),numel(dbPoints));
end
cI = repmat([1 -1],1,cfg.N/2);
cC = [ones(1,cfg.M) -ones(1,cfg.M)];
for r = 1:numel(rhoBValues)
    % Samples one bit apart are N AR(1) steps apart: rhoB = rhoS^N.
    rhoS = rhoBValues(r)^(1/cfg.N);
    if rhoBValues(r)==1, rhoS=1; end
    [g,h0,h1] = ar1_fading_bank(rhoS,cfg.timeTheoryTrials,cfg);
    for s = 1:numel(dbPoints)
        gamma = 10^(dbPoints(s)/10);
        time.I_CVNM(r,s) = cvnm_conditional_saddlepoint( ...
            g,gamma,cI,cfg,cfg.delta);
        time.contiguous(r,s) = cvnm_conditional_saddlepoint( ...
            g,gamma,cC,cfg,cfg.delta);
        time.NC_NoiseMod(r,s) = cvnm_conditional_saddlepoint( ...
            g,gamma,cC,cfg,cfg.deltaNc);
        [time.OODN(r,s),time.OODN_tau(r,s)] = ...
            optimized_statistical_threshold(g,gamma,'oodn',cfg);
        [time.NoiseMod(r,s),time.NoiseMod_tau(r,s)] = ...
            optimized_statistical_threshold(g,gamma,'noisemod',cfg);
        time.DBN(r,s) = dbn_conditional_saddlepoint(h0,h1,gamma,cfg);
    end
end
end

function [g,h0All,h1All] = ar1_fading_bank(rho,trials,cfg)
% Each row is an independent stationary two-bit window. Energy detectors use
% the first bit. DBN needs the complex gains over both bits, with no reset
% between the last sample of h0 and the first sample of h1.
g = zeros(trials,cfg.N);
h0All = complex(zeros(trials,cfg.N));
h1All = complex(zeros(trials,cfg.N));
batch = 1e4;
innovationScale = sqrt(max(0,1-rho^2));
done = 0;
while done < trials
    q = min(batch,trials-done);
    h = complex(zeros(q,2*cfg.N));
    h(:,1) = (randn(q,1)+1i*randn(q,1))/sqrt(2);
    for n = 2:2*cfg.N
        innovation = (randn(q,1)+1i*randn(q,1))/sqrt(2);
        h(:,n) = rho*h(:,n-1)+innovationScale*innovation;
    end
    h0 = h(:,1:cfg.N);
    h1 = h(:,cfg.N+1:end);
    rows = done+(1:q);
    g(rows,:) = abs(h0).^2;
    h0All(rows,:) = h0;
    h1All(rows,:) = h1;
    done = done+q;
end
end

function pe = cvnm_conditional_saddlepoint(g,gamma,code,cfg,delta)
% Conditioned on the gains, each sample energy is its variance times a
% unit-mean exponential. Multiplying the statistic by d makes either bit
% decision wrong on the same event, d*T < 0.
probability = zeros(size(g,1),1);
for d = [-1 1]
    receivedVariance = 1+gamma/cfg.N*g.*(1+d*delta*code);
    weights = d*code.*receivedVariance;
    probability = probability+0.5*saddlepoint_cdf(weights,0);
end
pe = mean(probability);
end

function pe = dbn_conditional_saddlepoint(h0,h1,gamma,cfg)
% The real part of the reference/data correlation contributes one positive
% and one negative eigenvalue per sample pair; both enter the CDF below.
a = gamma*cfg.dbnIntervalEnergy/cfg.N;
g0 = abs(h0).^2;
g1 = abs(h1).^2;
m = a*real(conj(h1).*h0);
determinantTerm = 1+a*(g0+g1);
root = sqrt(m.^2+determinantTerm);
lambdaPlus = 0.5*(m+root);
lambdaMinus = 0.5*(m-root);
pe = mean(saddlepoint_cdf([lambdaPlus lambdaMinus],0));
end

function probability = saddlepoint_cdf(weights,threshold)
% Lugannani--Rice lower-tail probability for each row of exponential weights.
meanValue = sum(weights,2);
% The nearest poles on either side of zero bound the interval where the
% cumulant-generating function exists. Stay inside them during bisection.
reciprocal = zeros(size(weights));
nonzero = weights~=0;
reciprocal(nonzero) = 1./weights(nonzero);
lowerCandidates = reciprocal;
lowerCandidates(weights>=0) = -Inf;
upperCandidates = reciprocal;
upperCandidates(weights<=0) = Inf;
lower = max(lowerCandidates,[],2);
upper = min(upperCandidates,[],2);
lower(~isfinite(lower)) = -1e6;
upper(~isfinite(upper)) = 1e6;
finiteLower = isfinite(lower) & lower>-1e6;
finiteUpper = isfinite(upper) & upper<1e6;
lower(finiteLower) = lower(finiteLower)*(1-1e-11);
upper(finiteUpper) = upper(finiteUpper)*(1-1e-11);
left = zeros(size(meanValue));
right = zeros(size(meanValue));
above = meanValue>threshold;
left(above)=lower(above); right(above)=0;
left(~above)=0; right(~above)=upper(~above);
for iteration = 1:52
    saddle = 0.5*(left+right);
    derivative = sum(weights./(1-saddle.*weights),2);
    below = derivative<threshold;
    left(below)=saddle(below);
    right(~below)=saddle(~below);
end
saddle = 0.5*(left+right);
denominator = 1-saddle.*weights;
cumulant = -sum(log(denominator),2);
second = sum((weights./denominator).^2,2);
w = sign(saddle).*sqrt(max(2*(saddle*threshold-cumulant),realmin));
u = saddle.*sqrt(second);
normalDensity = exp(-0.5*w.^2)/sqrt(2*pi);
probability = 0.5*erfc(-w/sqrt(2))+normalDensity.*(1./w-1./u);
% Near the mean both w and u approach zero. Use a local normal approximation
% rather than subtracting two nearly equal, large reciprocals.
nearMean = abs(saddle)<1e-7;
if any(nearMean)
    variance = sum(weights(nearMean,:).^2,2);
    z = (threshold-meanValue(nearMean))./sqrt(variance);
    probability(nearMean) = 0.5*erfc(-z/sqrt(2));
end
probability = min(max(probability,0),1);
end

function [pe,tau] = optimized_statistical_threshold(g,gamma,scheme,cfg)
% Fit one threshold per operating point, not one per fade. For conditional
% sample variances v_n = 1 + a*g_n, total energy has mean sum(v_n) and
% variance sum(v_n.^2). Match these moments before averaging the bit errors.
sumG = sum(g,2);
sumG2 = sum(g.^2,2);
switch scheme
    case 'oodn'
        a0 = 0;
        a1 = 2*gamma/cfg.N;
        upper = cfg.N+4*gamma;
    case 'noisemod'
        a0 = 2*gamma/(cfg.N*(1+cfg.noiseModAlpha));
        a1 = cfg.noiseModAlpha*a0;
        upper = cfg.N+2*cfg.noiseModAlpha*gamma/(1+cfg.noiseModAlpha);
    otherwise
        error('Unknown absolute-energy scheme.');
end
mean0 = cfg.N+a0*sumG;
variance0 = cfg.N+2*a0*sumG+a0^2*sumG2;
mean1 = cfg.N+a1*sumG;
variance1 = cfg.N+2*a1*sumG+a1^2*sumG2;
objective = @(threshold) statistical_threshold_objective( ...
    threshold,mean0,variance0,mean1,variance1);
options = optimset('TolX',2e-6,'Display','off');
[tau,pe] = fminbnd(objective,0.4*cfg.N,upper,options);
end

function pe = statistical_threshold_objective(tau,mean0,var0,mean1,var1)
falseAlarm = 1-moment_matched_energy_cdf(tau,mean0,var0);
miss = moment_matched_energy_cdf(tau,mean1,var1);
pe = 0.5*mean(falseAlarm+miss);
end

function probability = moment_matched_energy_cdf(tau,meanValue,variance)
shape = meanValue.^2./variance;
scale = variance./meanValue;
probability = gammainc(tau./scale,shape,'lower');
end

function time = mc_time_selective_all(rhoBValues,dbPoints,theory,cfg,seed)
% Draw fresh fades for the markers. Keep the OODN and NoiseMod thresholds
% obtained from the separate fading average; do not refit them on these bits.
rng(seed,'twister');
rhoBValues = rhoBValues(:).';
dbPoints = dbPoints(:).';
fields = {'I_CVNM','contiguous','NC_NoiseMod','OODN','NoiseMod','DBN'};
for f = 1:numel(fields)
    time.(fields{f}) = zeros(numel(rhoBValues),numel(dbPoints));
end
cI = repmat([1 -1],1,cfg.N/2);
cC = [ones(1,cfg.M) -ones(1,cfg.M)];
batch = 1e4;
for r = 1:numel(rhoBValues)
    errors = zeros(numel(fields),numel(dbPoints));
    rho = rhoBValues(r)^(1/cfg.N);
    if rhoBValues(r)==1, rho=1; end
    innovationScale = sqrt(max(0,1-rho^2));
    done = 0;
    lastDecile = 0;
    while done < cfg.timeTrials
        q = min(batch,cfg.timeTrials-done);
        h = complex(zeros(q,2*cfg.N));
        h(:,1) = (randn(q,1)+1i*randn(q,1))/sqrt(2);
        for n = 2:2*cfg.N
            innovation = (randn(q,1)+1i*randn(q,1))/sqrt(2);
            h(:,n) = rho*h(:,n-1)+innovationScale*innovation;
        end
        h0 = h(:,1:cfg.N);
        h1 = h(:,cfg.N+1:end);
        g = abs(h0).^2;
        for s = 1:numel(dbPoints)
            gamma = 10^(dbPoints(s)/10);
            d = 2*double(rand(q,1)>=0.5)-1;
            varianceI = 1+gamma/cfg.N*g.*(1+d*cfg.delta*cI);
            % Given g, -log(U) gives the exponential energy of one sample
            % in units of its received variance; its phase is not needed.
            statisticI = sum(cI.*varianceI.* ...
                (-log(max(rand(q,cfg.N),realmin))),2);
            errors(1,s) = errors(1,s)+sum(d.*statisticI<0);

            dC = 2*double(rand(q,1)>=0.5)-1;
            varianceC = 1+gamma/cfg.N*g.*(1+dC*cfg.delta*cC);
            statisticC = sum(cC.*varianceC.* ...
                (-log(max(rand(q,cfg.N),realmin))),2);
            errors(2,s) = errors(2,s)+sum(dC.*statisticC<0);

            dNc = 2*double(rand(q,1)>=0.5)-1;
            varianceNc = 1+gamma/cfg.N*g.*(1+dNc*cfg.deltaNc*cC);
            statisticNc = sum(cC.*varianceNc.* ...
                (-log(max(rand(q,cfg.N),realmin))),2);
            errors(3,s) = errors(3,s)+sum(dNc.*statisticNc<0);

            bitO = rand(q,1)>=0.5;
            varianceO = 1+2*gamma/cfg.N*g.*double(bitO);
            energyO = sum(varianceO.*(-log(max(rand(q,cfg.N),realmin))),2);
            errors(4,s) = errors(4,s)+sum( ...
                (energyO>=theory.OODN_tau(r,s))~=bitO);

            bitN = rand(q,1)>=0.5;
            a0 = 2*gamma/(cfg.N*(1+cfg.noiseModAlpha));
            coefficient = 1+(cfg.noiseModAlpha-1)*double(bitN);
            varianceN = 1+a0*g.*coefficient;
            energyN = sum(varianceN.*(-log(max(rand(q,cfg.N),realmin))),2);
            errors(5,s) = errors(5,s)+sum( ...
                (energyN>=theory.NoiseMod_tau(r,s))~=bitN);

            dB = 2*double(rand(q,1)>=0.5)-1;
            % Unlike the energy detectors, DBN needs phase. Reuse u across
            % the adjacent blocks but draw independent receiver noises.
            u = sqrt(gamma*cfg.dbnIntervalEnergy/cfg.N/2)* ...
                (randn(q,cfg.N)+1i*randn(q,cfg.N));
            w0 = (randn(q,cfg.N)+1i*randn(q,cfg.N))/sqrt(2);
            w1 = (randn(q,cfg.N)+1i*randn(q,cfg.N))/sqrt(2);
            y0 = h0.*u+w0;
            y1 = dB.*h1.*u+w1;
            statisticB = real(sum(conj(y1).*y0,2));
            errors(6,s) = errors(6,s)+sum(dB.*statisticB<0);
        end
        done = done+q;
        decile = floor(10*done/cfg.timeTrials);
        if decile > lastDecile
            fprintf('  rho_b=%g (%d/%d): %d/%d trials per SNR and scheme.\n', ...
                rhoBValues(r),r,numel(rhoBValues),done,cfg.timeTrials);
            lastDecile = decile;
        end
    end
    for f = 1:numel(fields)
        time.(fields{f})(r,:) = errors(f,:)/done;
    end
end
end

function single_frontend(preset,outDir)
% Fig. 2(c): test one receiver impairment at a time, without compensation.
% Both code placements give the same two group energies in distribution on
% this unit-gain AWGN link. It suffices to count how often E_high < E_low.
M=32; N=2*M; dcs=[.1 .5 1]; irrs=[20 10 5];
names={'Ideal','DC01','DC05','DC1','IQ20','IQ10','IQ5'};
snrs=0:21; dense=(0:210)/10; bits=20000000;
if strcmp(preset,'smoke')
    bits=2000; snrs=[0 5 10 16 21]; dense=unique([0:2:20 21]);
end
prior=rng; restoreRng=onCleanup(@() rng(prior)); %#ok<NASGU>
[u,w]=sw_beta_quad(48,M/2-1);
% asym is the image/direct amplitude ratio. Unit total I/Q gain gives
% power gains 1+kappa and 1-kappa along the two principal axes.
pe=zeros(numel(dense),numel(names)); asym=10.^(-irrs/20);
kappa=2*asym./(1+asym.^2);
for i=1:numel(dense)
    vl=10^(-dense(i)/10); vh=vl+2/N; z=vl/(vl+vh);
    pe(i,1)=betainc(z,M,M);
    for j=1:3
        % Given the Poisson counts, both offset group energies are Gamma;
        % the beta CDF gives the conditional probability that their order flips.
        ph=sw_poisson_weights(M*dcs(j)/(N*vh));
        pl=sw_poisson_weights(M*dcs(j)/(N*vl));
        [a,b]=ndgrid(M+(0:numel(ph)-1),M+(0:numel(pl)-1));
        pe(i,j+1)=ph.'*betainc(repmat(z,size(a)),a,b)*pl;
        % u is the fraction of a group's ideal energy in the first I/Q axis.
        % Integrate over that fraction independently for the high and low groups.
        c=1-kappa(j)+2*kappa(j)*u;
        argument=(vl*c.')./(vh*c+vl*c.');
        pe(i,j+4)=w.'*betainc(argument,M,M)*w;
    end
end
assert(all(isfinite(pe(:))) && all(pe(:)>0 & pe(:)<=0.5));
% Regression check for the theoretical curves at 16 dB (ideal, DC, then I/Q).
expected16=[.0007403040 .0011272605 .0037339648 .0092873461 ...
            .0008925876 .0027268240 .0076549177];
at16=find(abs(dense-16)<1e-10,1);
assert(~isempty(at16) && max(abs(pe(at16,:)-expected16))<1e-8, ...
    'The computed front-end BEPs failed their 16-dB reference check.');
writetable(array2table([dense(:),pe],'VariableNames',[{'EbN0_dB'},names]), ...
    fullfile(outDir,'impairment_exact.csv'));

seed=2026112103; rng(seed,'twister');
counts=zeros(numel(snrs),7); step=100000; done=0; decile=0;
while done<bits
    q=min(step,bits-done);
    % The same Gaussian modes drive all severities and SNRs in this batch;
    % 'h' and 'l' here label the high- and low-variance partitions.
    [hr,hi,zh]=sw_group_modes(q,M); [lr,li,zl]=sw_group_modes(q,M);
    hs=hr+hi; ls=lr+li;
    for j=1:numel(snrs)
        vl=10^(-snrs(j)/10); vh=vl+2/N;
        twiceT=vh*hs-vl*ls;
        counts(j,1)=counts(j,1)+nnz(twiceT<0);
        % The common offset adds M*|b|^2 to both groups. After subtraction
        % only its cross term remains; |b| = sqrt(D/N) since Eb = 1.
        cross=2*sqrt(2*M)*(sqrt(vh)*zh-sqrt(vl)*zl);
        for k=1:3
            counts(j,1+k)=counts(j,1+k)+nnz(twiceT+sqrt(dcs(k)/N)*cross<0);
            eH=(1+kappa(k))*hr+(1-kappa(k))*hi;
            eL=(1+kappa(k))*lr+(1-kappa(k))*li;
            counts(j,4+k)=counts(j,4+k)+nnz(vh*eH<vl*eL);
        end
    end
    done=done+q;
    if floor(10*done/bits)>decile
        decile=floor(10*done/bits); fprintf('  Fig. 2(c): %d/%d bits per point.\n',done,bits);
    end
end
writetable(array2table([snrs(:),counts/bits],'VariableNames',[{'EbN0_dB'},names]), ...
    fullfile(outDir,'impairment_mc.csv'));
[si,sj]=ndgrid((1:numel(snrs)).',1:7); [x,y]=deal(snrs(si(:)),names(sj(:)));
writetable(table(x(:),y(:),repmat(bits,numel(si),1),counts(:),counts(:)/bits, ...
    repmat(seed,numel(si),1),'VariableNames',{'EbN0_dB','Scheme','Bits','Errors','BEP','Seed'}), ...
    fullfile(outDir,'impairment_counts.csv'));
sw_tail_points(bits,M,outDir);
end

function [realE,imagE,z]=sw_group_modes(q,M)
% Rotate the real coordinates so the offset lies along the first mode, z.
% Keep z in realE: the DC cross term and the group energy share this draw.
z=randn(q,1);
realE=z.^2+2*sb_gamma((M-1)/2,q,1);
imagE=2*sb_gamma(M/2,q,1);
end

function p=sw_poisson_weights(lambda)
% Once the Poisson weights decrease, a geometric tail bound controls truncation.
p=exp(-lambda); k=0;
while true
    next=p(end)*lambda/(k+1);
    if k+2>lambda && next/(1-lambda/(k+2))<1e-28, break; end
    p(end+1,1)=next; k=k+1; %#ok<AGROW>
end
p=p(:);
end

function [u,w]=sw_beta_quad(n,alpha)
% Golub-Welsch quadrature for normalized Beta(alpha+1,alpha+1) on [0,1].
k=(1:n-1).'; off=sqrt(k.*(k+2*alpha)./((2*k+2*alpha).^2-1));
[V,D]=eig(diag(off,1)+diag(off,-1));
[nodes,order]=sort(diag(D)); u=(nodes+1)/2;
w=V(1,order).'.^2; w=w/sum(w);
end

function sw_tail_points(bits,M,outDir)
% The target BEPs determine the SNR grid, not the error counts. At each SNR
% draw the two independent Gamma group energies and count ordering errors.
N=2*M; targets=[1e-4 1e-5 2e-6 1e-6]; deltas=[9/11 1];
names={'NC_NoiseMod','CVNM'}; rows=8;
db=zeros(rows,1); E=zeros(rows,1); seeds=E; D=E; target=E;
tag=cell(rows,1); sparsePe=nan(rows,2); m=0;
for i=1:2
    delta=deltas(i);
    for j=1:numel(targets)
        m=m+1; target(m)=targets(j); D(m)=delta; tag{m}=names{i};
        z=betaincinv(target(m),M,M);
        gamma=N*(1-2*z)/(2*z+delta-1);
        db(m)=10*log10(gamma); vh=1/gamma+(1+delta)/N; vl=1/gamma+(1-delta)/N;
        seeds(m)=2026112139+m; rng(seeds(m),'twister'); done=0;
        while done<bits
            q=min(200000,bits-done);
            E(m)=E(m)+nnz(vh*sb_gamma(M,q,1)<vl*sb_gamma(M,q,1)); done=done+q;
        end
        sparsePe(m,3-i)=E(m)/bits;
    end
end
t=table(db,sparsePe(:,1),sparsePe(:,2),'VariableNames',{'EbN0_dB','CVNM','NC_NoiseMod'});
writetable(sortrows(t,'EbN0_dB'),fullfile(outDir,'tail_plot.csv'));
writetable(table(tag,D,db,target,repmat(bits,rows,1),E,E/bits,seeds, ...
    'VariableNames',{'Scheme','Delta','EbN0_dB','DesignBEP','Bits','Errors','BEP','Seed'}), ...
    fullfile(outDir,'tail_counts.csv'));
end

function outDir = run_fig3c(preset,outDir)
% Fig. 3(c): compare code order at the same N, contrast, bit and channel.
if nargin < 1 || isempty(preset), preset = 'full'; end
preset = validatestring(preset,{'full','smoke'});
if nargin < 2 || isempty(outDir)
    outDir = fullfile(pwd,'runs',['fig3c_' preset '_' datestr(now,'yyyymmdd_HHMMSSFFF')]);
end
outDir = char(outDir);
if isfile(outDir) || isfolder(outDir)
    error('CVNM:OutputExists','Select a new directory. Existing output: %s',outDir);
end
[ok,msg] = mkdir(outDir); assert(ok,'%s',msg);

cfg.preset = preset; cfg.bits = 20000000;
if strcmp(preset,'smoke'), cfg.bits = 2000; end
cfg.lengths = [16,32,64,128]; cfg.contrasts = [0.25,0.5,9/11,1];
cfg.Eb = 1; cfg.EbN0_dB = 18; cfg.rho_b = 0.5;
cfg.batch = 32768; cfg.seedBase = 20263115; cfg.seeds = cfg.seedBase + cfg.lengths;
cfg.generator = 'mt19937ar'; cfg.matlabVersion = version;
fid = fopen(fullfile(outDir,'settings.json'),'wt'); assert(fid >= 0,'Cannot create settings.json.');
closer = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',jsonencode(cfg)); clear closer

nLen = numel(cfg.lengths); nDelta = numel(cfg.contrasts);
long = zeros(nLen*nDelta,14); ratios = nan(nLen,nDelta);
noise = cfg.Eb/10^(cfg.EbN0_dB/10);
% Hold the bit-spaced correlation fixed as N changes, not the sample correlation.
for i = 1:nLen
    N = cfg.lengths(i); rho = cfg.rho_b^(1/N); beta = sqrt(1-rho^2);
    contiguous = [ones(1,N/2),-ones(1,N/2)];
    alternating = (-1).^(0:(N-1));
    rs = RandStream(cfg.generator,'Seed',cfg.seeds(i));
    err = zeros(3,nDelta); done = 0; lastProgress = 0;
    fprintf('Fig. 3(c), N=%d, %d bits per code and contrast.\n',N,cfg.bits);
    while done < cfg.bits
        m = min(cfg.batch,cfg.bits-done);
        d = 2*(rand(rs,m,1) >= .5)-1;
        qc = zeros(m,1); qi = zeros(m,1); spread = zeros(m,1);
        for j = 1:N
            z = complex(randn(rs,m,1),randn(rs,m,1))/sqrt(2);
            % Start each bit in stationarity, then evolve the channel within it.
            if j == 1, h = z; else, h = rho*h + beta*z; end
            e = -log(max(realmin,rand(rs,m,1)));   % Exp(1), shared in the pair
            ge = (cfg.Eb/N)*abs(h).^2.*e;
            base = noise*e + ge;
            qc = qc + contiguous(j)*base; qi = qi + alternating(j)*base;
            spread = spread + ge;
        end
        % For either code, d*T = d*sum(c_j*base_j) + delta*spread because
        % c_j^2 = 1. Only the first term changes when we reorder the code.
        ec = (d.*qc + spread.*cfg.contrasts) < 0;
        ei = (d.*qi + spread.*cfg.contrasts) < 0;
        err = err + [sum(ec,1);sum(ei,1);sum(ec & ei,1)];
        done = done + m;
        progress = floor(10*done/cfg.bits);
        if progress > lastProgress
            fprintf('  N=%d: %d/%d bits\n',N,done,cfg.bits); lastProgress = progress;
        end
    end
    for j = 1:nDelta
        a = err(1,j); b = err(2,j); both = err(3,j);
        ratio = NaN; lo = NaN; hi = NaN;
        if b > 0, ratio = a/b; end
        if a > 0 && b > 0
            % 'both' counts joint errors on the same bits. The pointwise
            % 95% log-ratio interval uses this covariance, not independent runs.
            halfWidth = 1.959963984540054*sqrt(max(0,a+b-2*both)/(a*b));
            lo = ratio*exp(-halfWidth); hi = ratio*exp(halfWidth);
        end
        ratios(i,j) = ratio;
        long((i-1)*nDelta+j,:) = [N,cfg.contrasts(j),cfg.rho_b,cfg.EbN0_dB, ...
            cfg.bits,cfg.seeds(i),a,b,both,a/cfg.bits,b/cfg.bits,ratio,lo,hi];
    end
end
fields = {'N','delta','rho_b','ebn0_db','bits','seed','errors_contiguous', ...
    'errors_alternating','errors_both','bep_contiguous','bep_alternating', ...
    'ratio','paired_lo95','paired_hi95'};
writetable(array2table(long,'VariableNames',fields),fullfile(outDir,'paired_trials.csv'));
writetable(array2table([cfg.lengths(:),ratios], ...
    'VariableNames',{'N','D025','D050','D911','D100'}),fullfile(outDir,'order_gain.csv'));
fprintf('Fig. 3(c) complete: %s\n',outDir);
end

function [metrics, ratios] = summarize_fig3(aDir,bDir,cDir,outDir,bits)
%SUMMARIZE_FIG3 Match each BEP prediction to the marker at the same SNR and correlation.
% Panels (a,b) compare predictions with simulations; panel (c) compares code order.
if nargin<5 || isempty(bits), bits=20000000; end
validateattributes(bits,{'numeric'},{'real','scalar','integer','positive','finite'});
assert(~isfile(outDir) && ~isfolder(outDir),'Choose a new summary directory.');

cols={'OODN_statistical_threshold','NoiseMod_alpha10_statistical_threshold', ...
    'contiguous_CVNM','I_CVNM','NC_NoiseMod','DBN_L100'};
schemes={'OODN';'NoiseMod';'Contiguous CVNM';'I-CVNM';'NC-NoiseMod';'DBN'};
gridA=[0 2.5 5 7.5 10 12.5 15 16.5 18 19 20];
gridB=[.02 .035 .05 .075 .10 .15 .20 .275 .35 .425 .50 .575 .65 .725 ...
    .80 .85 .90 .935 .97 .985 1];
[ea,da,na]=check_sweep(aDir,'time_snr','EbN0_dB',cols,bits,gridA);
[eb,db,nb]=check_sweep(bDir,'time_rho','rho_b',cols,bits,gridB);
assert(all(da>0) && all(db>0),'NMSE is undefined for an all-zero MC reference.');
% Pool squared errors and squared reference BEPs before taking their ratio in dB.
% Averaging the two panel NMSEs in dB would give a different result.
% A shared physical operating point appears once in each displayed sweep.
metrics=table(schemes,repmat(na,6,1),repmat(nb,6,1), ...
    10*log10(ea./da)',10*log10(eb./db)',10*log10((ea+eb)./(da+db))', ...
    'VariableNames',{'Scheme','Markers_a','Markers_b','NMSE_a_dB','NMSE_b_dB','NMSE_pooled_dB'});
ratios=check_placements(cDir,bits);

[ok,msg]=mkdir(outDir); assert(ok,'%s',msg);
for suffix={'_mc.csv','_theory.csv'}
    copyfile(fullfile(aDir,['time_snr' suffix{1}]),outDir);
    copyfile(fullfile(bDir,['time_rho' suffix{1}]),outDir);
end
copyfile(fullfile(cDir,'order_gain.csv'),outDir);
writetable(metrics,fullfile(outDir,'nmse.csv'));
writetable(ratios,fullfile(outDir,'ratio_summary.csv'));
fprintf('%g decisions/scheme/point; %d SNR and %d correlation markers/scheme.\n',bits,na,nb);
fprintf('Paired contiguous/interleaved BEP ratios range from %.4f to %.4f.\n',min(ratios.ratio),max(ratios.ratio));
disp(metrics(:,{'Scheme','NMSE_pooled_dB'}));
end

function [e,d,n]=check_sweep(folder,stem,axis,cols,bits,grid)
mc=readtable(fullfile(folder,[stem '_mc.csv']));
th=readtable(fullfile(folder,[stem '_theory.csv']));
ct=readtable(fullfile(folder,[stem '_error_counts.csv']));
x=mc.(axis); tx=th.(axis); cx=ct.(axis); n=height(mc);
assert(n>0 && numel(unique(x))==n && all(isfinite(x)));
assert(all(isfinite(tx)) && numel(unique(tx))==height(th),'Invalid theoretical grid.');
assert(numel(unique(cx))==height(ct) && height(ct)==n && all(ct.BitsPerScheme==bits), ...
    'Unexpected Monte Carlo grid or trial count in %s.',stem);
if bits==20000000
    assert(n==numel(grid) && all(abs(sort(x)-sort(grid(:)))<1e-12),'Unexpected full grid in %s.',stem);
end
[hasC,idxC]=ismember(x,cx); [hasT,idxT]=ismember(x,tx);
assert(all(hasC) && all(hasT),'Missing count or theoretical point in %s.',stem);
p=mc{:,cols}; pTheory=th{idxT,cols}; counts=ct{idxC,cols};
assert(all(isfinite(p)&p>=0&p<=1,'all') && all(isfinite(pTheory)&pTheory>=0&pTheory<=1,'all'));
assert(all(counts==fix(counts)&counts>=0&counts<=bits,'all') && ...
    all(abs(p-counts./bits)<1e-12,'all'),'BEP and integer errors disagree in %s.',stem);
e=sum((pTheory-p).^2,1);
d=sum(p.^2,1);
end

function ratios=check_placements(folder,bits)
t=readtable(fullfile(folder,'paired_trials.csv'));
w=readtable(fullfile(folder,'order_gain.csv'));
ns=[16;32;64;128]; deltas=[.25 .5 9/11 1]; fields={'D025','D050','D911','D100'};
assert(height(t)==16 && height(w)==4 && isequal(sort(w.N),ns));
assert(all(t.bits==bits) && all(abs(t.rho_b-.5)<1e-12) && all(abs(t.ebn0_db-18)<1e-12));
counts=[t.errors_contiguous,t.errors_alternating,t.errors_both];
assert(all(isfinite(counts)&counts>=0&counts<=bits&counts==fix(counts),'all'));
assert(all(counts(:,3)<=min(counts(:,1:2),[],2)) && ...
    all(sum(counts(:,1:2),2)-counts(:,3)<=bits),'Invalid paired error counts.');
assert(all(t.errors_alternating>0),'The BEP ratio is undefined at zero denominator errors.');
r=t.errors_contiguous./t.errors_alternating;
assert(all(abs(t.bep_contiguous-t.errors_contiguous/bits)<1e-12) && ...
    all(abs(t.bep_alternating-t.errors_alternating/bits)<1e-12));
assert(all(abs(r-t.ratio)<=1e-10*max(1,abs(r))),'Long-form ratios disagree with counts.');
for i=1:numel(ns)
    for j=1:numel(deltas)
        k=find(t.N==ns(i) & abs(t.delta-deltas(j))<1e-12);
        assert(isscalar(k),'Missing or duplicated panel (c) comparison.');
        q=w{w.N==ns(i),fields(j)};
        assert(abs(q-r(k))<=1e-10*max(1,abs(r(k))),'Plotted ratio disagrees with counts.');
    end
end
ratios=sortrows(t(:,{'N','delta','bits','errors_contiguous','errors_alternating', ...
    'bep_contiguous','bep_alternating','ratio'}),{'N','delta'});
end

function single_plot(outDir,preset)
% Both a new simulation and a redraw read the saved tables. Redrawing therefore
% preserves the Monte Carlo markers, including the low-BEP points.
    nominal = fullfile(outDir,'nominal'); front = fullfile(outDir,'frontend');
    order = fullfile(outDir,'order');
    % Same colors and markers in both figures. Times is used when installed.
    pal = [22 22 22;170 82 127;113 68 132;0 124 99;186 84 15;0 108 156;90 98 108;0 117 143]/255;
    th = readtable(fullfile(nominal,'nominal_theory.csv'));
    am = readtable(fullfile(nominal,'awgn_mc.csv'));
    rm = readtable(fullfile(nominal,'ray_mc.csv'));
    tail = readtable(fullfile(front,'tail_plot.csv'));
    it = readtable(fullfile(front,'impairment_exact.csv'));
    im = readtable(fullfile(front,'impairment_mc.csv'));
    f = swplot_figure('Figure 2',preset);
    cleanup2 = onCleanup(@() close(f)); %#ok<NASGU>
    ax = swplot_axes(f,1,'E_b/N_0 (dB)',[0 22],[0 5 10 15 20 22],1e-6);
    y1 = {'CVNM_AWGN_exact','NC_NoiseMod_AWGN_exact','DBN_AWGN_exact','OODN_AWGN_exact','NoiseMod_AWGN_exact'};
    m1 = {'CVNM','NC_NoiseMod','DBN','OODN','NoiseMod'};
    labels = {'CVNM/I-CVNM','NC-NoiseMod, \alpha=10','DBN','OODN','NoiseMod, \alpha=10'};
    h = swplot_series(ax,th,am,'EbN0_dB',y1,m1,pal([1 3 6 4 5],:),{'o','+','d','s','^'},'-');
    for k = 1:2
        swplot_mark(ax,tail.EbN0_dB,tail.(m1{k}),h(k).Color,h(k).Marker,true);
    end
    swplot_text(ax,.975,.96,'(a) AWGN','right');
    swplot_key(ax,h,labels,1,'sw');

    ax = swplot_axes(f,2,'E_b/N_0 (dB)',[0 30],0:5:30,.001);
    y2 = {'CVNM_Rayleigh_exact','NC_NoiseMod_Rayleigh_exact','DBN_Rayleigh_exact', ...
        'OODN_Rayleigh_perfect_CSI','OODN_Rayleigh_fixed','NoiseMod_Rayleigh_fixed'};
    m2 = {'CVNM','NC_NoiseMod','DBN','OODN_perfect_CSI','OODN_fixed','NoiseMod_fixed'};
    labels = {'CVNM/I-CVNM, no CSI','NC-NoiseMod, \alpha=10','DBN, pair-coherent', ...
        'OODN, perfect CSI','OODN, fixed thr.','NoiseMod, fixed thr.'};
    h = swplot_series(ax,th,rm,'EbN0_dB',y2,m2,pal([1 3 6 7 4 5],:),{'o','+','d','v','s','^'},'-');
    swplot_text(ax,.975,.96,'(b) Flat Rayleigh fading','right');
    swplot_key(ax,h,labels,1,'sw');

    ax = swplot_axes(f,3,'E_b/N_0 (dB)',[0 21],[0 5 10 15 20 21],1e-6);
    ci = {'Ideal','DC01','DC05','DC1','IQ20','IQ10','IQ5'};
    labels = {'Ideal','DC, D=0.1','DC, D=0.5','DC, D=1','I/Q, IRR=20 dB','I/Q, IRR=10 dB','I/Q, IRR=5 dB'};
    h = swplot_series(ax,it,im,'EbN0_dB',ci,ci,pal([1 2 5 3 6 4 7],:),{'o','+','d','^','x','s','v'},'-');
    swplot_text(ax,.025,.75,{'(c) CVNM/I-CVNM:','receiver impairments'},'left');
    swplot_key(ax,h,labels,1,'sw');
    swplot_export(f,outDir,'Fig2');

    f3 = swplot_figure('Figure 3',preset);
    cleanup3 = onCleanup(@() close(f3)); %#ok<NASGU>
    fields = {'I_CVNM','contiguous_CVNM','NC_NoiseMod','OODN_statistical_threshold', ...
        'NoiseMod_alpha10_statistical_threshold','DBN_L100'};
    cols = pal([1 2 3 4 5 6],:);
    mk = {'o','x','+','s','^','d'};
    labels = {'I-CVNM','Contiguous CVNM','NC-NoiseMod, \alpha=10','OODN','NoiseMod, \alpha=10','DBN'};
    for k=1:2
        if k==1
            base='time_snr'; xcol='EbN0_dB'; xl='E_b/N_0 (dB)';
            lim=[-.35 20.35]; ticks=0:5:20; titleTxt='(a) \rho_b=0.5'; side='sw';
        else
            base='time_rho'; xcol='rho_b'; xl='Symbol-spaced correlation, \rho_b';
            lim=[-.015 1.018]; ticks=0:.2:1; titleTxt='(b) E_b/N_0=18 dB'; side='se';
        end
        a = readtable(fullfile(nominal,[base '_theory.csv']));
        b = readtable(fullfile(nominal,[base '_mc.csv']));
        ax = swplot_axes(f3,k,xl,lim,ticks,.001);
        h = swplot_series(ax,a,b,xcol,fields,fields,cols,mk,':');
        swplot_text(ax,.982,.98,titleTxt,'right');
        ix = [1 4 6 2 3 5]; % column-major display groups related entries
        swplot_key(ax,h(ix),labels(ix),2,side);
    end
    ratio = readtable(fullfile(order,'order_gain.csv'));
    cs = {'D025','D050','D911','D100'}; cols=pal([1 8 5 3],:); mk={'o','x','d','^'};
    ax = swplot_axes(f3,3,'Samples per bit, N',[14.3 144],[16 32 64 128],0);
    set(ax,'XScale','log','XMinorTick','off','XMinorGrid','off','XTickLabel',{'16','32','64','128'});
    ylabel(ax,'BEP ratio');
    yr=ratio{:,cs}; finite=yr(isfinite(yr));
    if isempty(finite), lo=.9; hi=2.55;
    else, lo=min(.9,min(finite)-.04); hi=max(2.55,max(finite)+.15); end
    ylim(ax,[lo hi]);
    if lo==.9 && hi==2.55, yticks(ax,1:.2:2.4); end
    plot(ax,xlim(ax),[1 1],'--','Color',[.5 .5 .5],'LineWidth',.65,'HandleVisibility','off');
    h=gobjects(4,1);
    for j=[1 4 3 2]
        h(j)=swplot_mark(ax,ratio.N,ratio.(cs{j}),cols(j,:),mk{j},false);
    end
    swplot_text(ax,.982,.98,'(c) \rho_b=0.5, E_b/N_0=18 dB','right');
    labels={'\delta=0.25','\delta=0.5','\delta=9/11','\delta=1'}; ix=[1 3 2 4];
    swplot_key(ax,h(ix),labels(ix),2,'ne');
    swplot_export(f3,outDir,'Fig3');
end

function f=swplot_figure(name,preset)
    f=figure('Visible','off','Color','w','Name',[name ' (' char(preset) ')'], ...
        'NumberTitle','off','Units','inches','Position',[.2 .2 9.1 2.6], ...
        'PaperPositionMode','auto','Renderer','painters');
end

function ax=swplot_axes(f,k,xlab,xlim0,tick0,ymin)
    left=[.063 .403 .734]; 
    ax=axes('Parent',f,'Units','normalized','Position',[left(k) .205 .253 .755], ...
        'FontName','Times New Roman','FontSize',8.2,'Box','off','LineWidth',.65, ...
        'TickDir','out','TickLength',[.015 .015],'Color','white','Layer','bottom', ...
        'XGrid','on','YGrid','on','GridLineStyle',':','MinorGridLineStyle',':', ...
        'GridColor',[.25 .25 .25],'GridAlpha',.35,'MinorGridColor',[.4 .4 .4],'MinorGridAlpha',.22);
    hold(ax,'on'); xlim(ax,xlim0); xticks(ax,tick0);
    xlabel(ax,xlab,'FontSize',9,'Interpreter','tex');
    if ymin>0
        set(ax,'YScale','log','YLim',[ymin .6],'YTick',10.^(round(log10(ymin)):-1), ...
            'YMinorTick','on','YMinorGrid','on');
        ylabel(ax,'BEP','FontSize',9);
    end
end

function h=swplot_series(ax,th,mc,xname,tfields,mfields,colors,marks,lineStyle)
    n=numel(tfields); h=gobjects(n,1);
    for j=1:n
        % Nonpositive entries are not representable on the log scale. Keep
        % gaps instead of mapping zeros to an invented numerical floor.
        y=th.(tfields{j}); y(~isfinite(y) | y<=0)=NaN;
        plot(ax,th.(xname),y,'LineStyle',lineStyle,'Color',colors(j,:), ...
            'LineWidth',1.0,'HandleVisibility','off');
    end
    for j=1:n
        h(j)=swplot_mark(ax,mc.(xname),mc.(mfields{j}),colors(j,:),marks{j},true);
    end
end

function h=swplot_mark(ax,x,y,color,marker,positiveOnly)
    keep=isfinite(x)&isfinite(y); if positiveOnly, keep=keep & y>0; end
    h=plot(ax,x(keep),y(keep),'LineStyle','none','Marker',marker,'Color',color, ...
        'MarkerEdgeColor',color,'MarkerFaceColor','white','MarkerSize',4.3,'LineWidth',.85);
end

function swplot_text(ax,x,y,s,align)
    text(ax,x,y,s,'Units','normalized','Interpreter','tex','FontName','Times New Roman', ...
        'FontSize',8.5,'VerticalAlignment','top','HorizontalAlignment',align, ...
        'BackgroundColor','white','Margin',.5);
end

function swplot_key(ax,h,labels,columns,side)
    key=legend(ax,h,labels,'Location','southwest','Orientation','vertical','NumColumns',columns, ...
        'FontName','Times New Roman','FontSize',7.1,'Interpreter','tex', ...
        'AutoUpdate','off','Color','white','EdgeColor',[.6 .6 .6],'Box','on');
    key.ItemTokenSize=[10 7];
    drawnow;
    key.Units='normalized'; a=ax.Position; p=key.Position;
    if strcmp(side,'sw'), p(1)=a(1)+.016*a(3);
    else, p(1)=a(1)+.982*a(3)-p(3); end
    if strcmp(side,'ne'), p(2)=a(2)+.865*a(4)-p(4);
    else, p(2)=a(2)+.022*a(4); end
    key.Position=p;
end

function swplot_export(f,outDir,stem)
    drawnow;
    savefig(f,fullfile(outDir,[stem '.fig']));
    exportgraphics(f,fullfile(outDir,[stem '.pdf']),'ContentType','vector','BackgroundColor','white');
end

