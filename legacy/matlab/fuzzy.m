%% Step 1: Generate fake dataset
clc; clear; rng(42);

N = 1000;

HR = 60 + (120-60).*rand(N,1);    % Heart Rate
Temp = 36 + (40-36).*rand(N,1);   % Temperature

% HealthStatus
HealthStatus = zeros(N,1);
for i = 1:N
    if HR(i) <= 90 && Temp(i) <= 37.5
        HealthStatus(i) = 1;
    elseif HR(i) <= 105 && Temp(i) <= 38.5
        HealthStatus(i) = 2;
    else
        HealthStatus(i) = 3;
    end
end

X = [HR Temp]; % Inputs

%% Step 2: Create FIS using old syntax
fis = newfis('HealthFIS','mamdani');

% Input HR
fis = addvar(fis,'input','HR',[60 120]);
fis = addmf(fis,'input',1,'Low','trimf',[60 60 90]);
fis = addmf(fis,'input',1,'Normal','trimf',[80 95 110]);
fis = addmf(fis,'input',1,'High','trimf',[100 120 120]);

% Input Temp
fis = addvar(fis,'input','Temp',[36 40]);
fis = addmf(fis,'input',2,'Low','trimf',[36 36 37.5]);
fis = addmf(fis,'input',2,'Normal','trimf',[37 37.8 38.5]);
fis = addmf(fis,'input',2,'High','trimf',[38 40 40]);

% Output HealthStatus
fis = addvar(fis,'output','HealthStatus',[1 3]);
fis = addmf(fis,'output',1,'Healthy','trimf',[1 1 2]);
fis = addmf(fis,'output',1,'Moderate','trimf',[1.5 2 2.5]);
fis = addmf(fis,'output',1,'Critical','trimf',[2 3 3]);

% Rules
ruleList = [
    1 1 1 1 1;
    2 2 2 1 1;
    3 3 3 1 1;
    2 1 2 1 1;
    1 2 2 1 1;
    3 2 3 1 1;
    2 3 2 1 1;
    1 3 2 1 1;
    3 1 2 1 1;
];
fis = addrule(fis,ruleList);

%% Step 3: Evaluate FIS
pred = evalfis(X,fis);   % íÌÈ Ãä íÚãá ÇáÂä
predClass = round(pred);

acc = sum(predClass == HealthStatus)/N;
fprintf('Baseline FIS Accuracy: %.2f%%\n', acc*100);
