clear all; close all; clc;

%UCITAVANJE TABELE
T = readtable('RSSI Database.xls', 'VariableNamingRule','preserve');

%INDEKSI PARAMETARA I KOORDINATA
apCols = 1:8;
xyCols = 9:10;

%KONSTANTE
noiseFloor = -120;
metersPerUnit = 0.1;
N = height(T);
nLoc = N/4;



% KONVERZIJA * U -120dBm
AP = zeros(N, numel(apCols));
for j = 1:numel(apCols)
    col = T{:, apCols(j)};
    if iscell(col) || isstring(col) || ischar(col)
        col = string(col);
        col(col=="*") = string(noiseFloor);
        col = str2double(col);
    end
    col(isnan(col)) = noiseFloor;
    AP(:,j) = col;
end

% XY KOORDINATE
XY = T{:, xyCols};
if iscell(XY); XY = str2double(string(XY)); end

%UZET PROSEK 4 REDA
X = zeros(nLoc, size(AP,2));
Y_pix = zeros(nLoc, 2);
for g = 1:nLoc
    idx = (g-1)*4 + (1:4);
    X(g,:)     = mean(AP(idx,:), 1);
    Y_pix(g,:) = XY(idx(1),:);
end
Y_m = Y_pix * metersPerUnit;

%NORMALIZACIJA MIN-MAX
[Xn, Xps] = mapminmax(X', -1, 1);
Yn = Y_m';

%NEURAL NET
hiddenlayers = [10 20];
net = feedforwardnet(hiddenlayers, 'trainlm');

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 80/100;
%net.divideParam.valRatio   = 0/100;
net.divideParam.testRatio  = 20/100;

[net, tr] = train(net, Xn, Yn);

%TESTIRANJE
Xtest = Xn(:, tr.testInd);
Ytrue = Yn(:, tr.testInd);
Yhat  = net(Xtest);

dx = Yhat(1,:) - Ytrue(1,:);
dy = Yhat(2,:) - Ytrue(2,:);
err = sqrt(dx.^2 + dy.^2);               

RMSE = sqrt(mean(err.^2));
MEAN = mean(err);
STD  = std(err);

fprintf('Test: RMSE=%.3f m | MEAN=%.3f m | STD=%.3f m | Ntest=%d\n', ...
        RMSE, MEAN, STD, numel(err));

%HISTOGRAM
figure; histogram(err, 30);
xlabel('Greška [m]'); ylabel('Broj tačaka');


%SWEEP BROJA NEURONA
rng(1);
sizes = 2:2:40;
rmse  = zeros(size(sizes));

for i = 1:numel(sizes)
    %JEDAN SKRIVENI SLOJ
    net = feedforwardnet(sizes(i), 'trainlm');

    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.80;
    net.divideParam.valRatio   = 0.00;
    net.divideParam.testRatio  = 0.20;

    net.trainParam.showWindow = false;

    [net, tr] = train(net, Xn, Yn);

    %GRESKA
    Xtest = Xn(:, tr.testInd);
    Ytrue = Yn(:, tr.testInd);
    Yhat  = net(Xtest);

    dx = Yhat(1,:) - Ytrue(1,:);
    dy = Yhat(2,:) - Ytrue(2,:);
    err = sqrt(dx.^2 + dy.^2);

    rmse(i) = sqrt(mean(err.^2));
end

%RMSE/NEURONI
figure;
plot(sizes, rmse, '-o');
xlabel('Broj neurona u skrivenom sloju');
ylabel('RMSE [m]');