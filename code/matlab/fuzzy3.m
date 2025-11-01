clc; clear;

%% -------------------------
% 1??  Ê·Ìœ »Ì«‰«  ÊÂ„Ì…
%% -------------------------
N = 100; % ⁄œœ «·⁄Ì‰« 
HR = 60 + (120-60)*rand(N,1);       % Heart Rate »Ì‰ 60 Ê 120
Temp = 36 + (40-36)*rand(N,1);     % Temperature »Ì‰ 36 Ê 40
X = [HR Temp];

% ≈‰‘«¡  ’‰Ì›«  HealthStatus (1=Healthy, 2=Moderate, 3=Critical)
Y = ones(N,1); 
for i=1:N
    if HR(i)>100 || Temp(i)>38
        Y(i)=3; % Critical
    elseif HR(i)>80 || Temp(i)>37.5
        Y(i)=2; % Moderate
    else
        Y(i)=1; % Healthy
    end
end

%% -------------------------
% 2?? ﬂ—Ê„Ê“Ê„ „»œ∆Ì (center,width,type ·ﬂ· MF)
%% -------------------------
% HR 3 MFs + Temp 3 MFs = 18 ﬁÌ„…
chrom = [65 20 1 90 15 3 115 20 2 36.5 1 1 37.5 1.5 3 39 1.5 2];

%% -------------------------
% 3?? œ«·… «··Ì«ﬁ… ··‹GA
%% -------------------------
fitnessFIS = @(chrom) myFitness(chrom,X,Y);

%% -------------------------
% 4?? ÕœÊœ «·„ €Ì—«  ··‹GA
%% -------------------------
lb = [40 1 1  80 1 1  100 1 1  36 0.5 1  37 0.5 1  38 0.5 1];
ub = [180 50 3  120 50 3  150 50 3  40 5 3  39.5 5 3  41 5 3];
nVars = 18;

%% -------------------------
% 5??  ‘€Ì· GA
%% -------------------------
options = optimoptions('ga','PopulationSize',30,'MaxGenerations',50,'Display','iter');
[bestChrom,fval] = ga(fitnessFIS, nVars, [],[],[],[], lb, ub, [], options);

disp('Best chromosome:');
disp(bestChrom);
disp(['Best fitness (error): ', num2str(fval)]);
disp(['Best accuracy: ', num2str((1-fval)*100),'%']);

%% =========================
% œ«·… «··Ì«ﬁ…
%% =========================
function error = myFitness(chrom,X,Y)
    % ≈‰‘«¡ FIS
    fis = newfis('HealthFIS','mamdani');

    % „œŒ· HR
    fis = addvar(fis,'input','HR',[40 180]);
    for i=1:3
        center = chrom((i-1)*3 +1);
        width  = chrom((i-1)*3 +2);
        type   = round(chrom((i-1)*3 +3));
        switch type
            case 1
                a=center-width/2; b=center; c=center+width/2;
                fis = addmf(fis,'input',1,'Low','trimf',[a b c]);
            case 2
                delta=width*0.2; a=center-width/2; b=center-delta; c=center+delta; d=center+width/2;
                fis = addmf(fis,'input',1,'Low','trapmf',[a b c d]);
            case 3
                sigma=width/2;
                fis = addmf(fis,'input',1,'Low','gaussmf',[sigma center]);
        end
    end

    % „œŒ· Temp
    fis = addvar(fis,'input','Temp',[35 42]);
    for i=1:3
        center = chrom(9 + (i-1)*3 +1);
        width  = chrom(9 + (i-1)*3 +2);
        type   = round(chrom(9 + (i-1)*3 +3));
        switch type
            case 1
                a=center-width/2; b=center; c=center+width/2;
                fis = addmf(fis,'input',2,'Low','trimf',[a b c]);
            case 2
                delta=width*0.2; a=center-width/2; b=center-delta; c=center+delta; d=center+width/2;
                fis = addmf(fis,'input',2,'Low','trapmf',[a b c d]);
            case 3
                sigma=width/2;
                fis = addmf(fis,'input',2,'Low','gaussmf',[sigma center]);
        end
    end

    % „Œ—ÛÃ HealthStatus
    fis = addvar(fis,'output','HealthStatus',[1 3]);
    fis = addmf(fis,'output',1,'Healthy','trimf',[1 1 2]);
    fis = addmf(fis,'output',1,'Moderate','trimf',[1.5 2 2.5]);
    fis = addmf(fis,'output',1,'Critical','trimf',[2 3 3]);

    % ﬁÊ«⁄œ fuzzy »”Ìÿ…
    ruleList = [
        1 1 1 1 1;
        2 2 2 1 1;
        3 3 3 3 1;
        2 1 2 1 1;
        1 2 2 1 1;
        3 2 3 2 1;
        2 3 2 2 1;
        1 3 2 2 1;
        3 1 2 2 1;
    ];
    fis = addrule(fis,ruleList);

    %  ﬁÌÌ„ FIS
    pred_cont = evalfis([X(:,1) X(:,2)],fis);
    pred = round(pred_cont);       
    pred(pred<1)=1; pred(pred>3)=3;

    % Õ”«» «·Œÿ√
    error = 1 - sum(pred==Y)/length(Y);
end
