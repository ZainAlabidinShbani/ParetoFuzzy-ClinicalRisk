clc; clear;

% -------------------------
%  Ê·Ìœ »Ì«‰«  ÊÂ„Ì…
% -------------------------
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

% ﬂ· MF: [center, width, type]
chrom = [65 20 1 90 15 3 115 20 2 36.5 1 1 37.5 1.5 3 39 1.5 2];

% -------------------------
% ≈‰‘«¡ FIS »«” Œœ«„ «·ÿ—Ìﬁ… «·ﬁœÌ„…
% -------------------------
fis = newfis('HealthFIS','mamdani');

% „œŒ· HR
fis = addvar(fis,'input','HR',[40 180]);
HR_MFs = {'Low','Normal','High'};
for i=1:3
    center = chrom( (i-1)*3 + 1 );
    width  = chrom( (i-1)*3 + 2 );
    type   = chrom( (i-1)*3 + 3 );
    
    switch type
        case 1 % Triangular
            a = center-width/2; b=center; c=center+width/2;
            fis = addmf(fis,'input',1,'Low','trimf',[a b c]); % «·«”„ ›ﬁÿ ··⁄—÷
        case 2 % Trapezoidal
            delta = width*0.2;
            a = center-width/2; b=center-delta; c=center+delta; d=center+width/2;
            fis = addmf(fis,'input',1,'Low','trapmf',[a b c d]);
        case 3 % Gaussian
            sigma = width/2;
            fis = addmf(fis,'input',1,'Low','gaussmf',[sigma center]);
    end
end

% „œŒ· Temp
fis = addvar(fis,'input','Temp',[35 42]);
Temp_MFs = {'Low','Normal','High'};
for i=1:3
    center = chrom(9 + (i-1)*3 + 1);
    width  = chrom(9 + (i-1)*3 + 2);
    type   = chrom(9 + (i-1)*3 + 3 );
    
    switch type
        case 1
            a = center-width/2; b=center; c=center+width/2;
            fis = addmf(fis,'input',2,'Low','trimf',[a b c]);
        case 2
            delta = width*0.2;
            a = center-width/2; b=center-delta; c=center+delta; d=center+width/2;
            fis = addmf(fis,'input',2,'Low','trapmf',[a b c d]);
        case 3
            sigma = width/2;
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

%  ﬁÌÌ„ ﬂ· «·⁄Ì‰« 
pred_cont = evalfis([HR Temp],fis);   % ﬁÌ„… „” „—…
pred = round(pred_cont);       
pred(pred<1)=1; pred(pred>3)=3;

% Õ”«» «·œﬁ…
accuracy = sum(pred==Y)/N * 100;
disp(['Accuracy of GA-optimized FIS: ', num2str(accuracy),'%']);
