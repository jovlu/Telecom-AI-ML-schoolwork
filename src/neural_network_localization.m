clear all; close all; clc;

% Load data
projectRoot = fileparts(fileparts(mfilename('fullpath')));
rssiFile = fullfile(projectRoot, 'data', 'RSSI Database.xls');
if ~isfile(rssiFile)
    error('Missing RSSI dataset. Expected file: %s', rssiFile);
end

T = readtable(rssiFile, 'VariableNamingRule','preserve');

% Feature and coordinate columns
apCols = 1:8;
xyCols = 9:10;

% Constants
noiseFloor = -120;
metersPerUnit = 0.1;
N = height(T);
nLoc = N/4;

% Convert * values to -120 dBm.
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

% XY coordinates
XY = T{:, xyCols};
if iscell(XY); XY = str2double(string(XY)); end

% Average four rows per measured location.
X = zeros(nLoc, size(AP,2));
Y_pix = zeros(nLoc, 2);
for g = 1:nLoc
    idx = (g-1)*4 + (1:4);
    X(g,:)     = mean(AP(idx,:), 1);
    Y_pix(g,:) = XY(idx(1),:);
end
Y_m = Y_pix * metersPerUnit;

% Min-max normalization
[Xn, Xps] = mapminmax(X', -1, 1);
Yn = Y_m';

% Neural network
hiddenlayers = [10 20];
net = feedforwardnet(hiddenlayers, 'trainlm');

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 80/100;
%net.divideParam.valRatio   = 0/100;
net.divideParam.testRatio  = 20/100;

[net, tr] = train(net, Xn, Yn);

% Evaluation
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

% Error histogram
figure; histogram(err, 30);
xlabel('Error [m]'); ylabel('Sample count');

% Hidden-neuron sweep
rng(1);
sizes = 2:2:40;
rmse  = zeros(size(sizes));

for i = 1:numel(sizes)
    % One hidden layer
    net = feedforwardnet(sizes(i), 'trainlm');

    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.80;
    net.divideParam.valRatio   = 0.00;
    net.divideParam.testRatio  = 0.20;

    net.trainParam.showWindow = false;

    [net, tr] = train(net, Xn, Yn);

    % Error
    Xtest = Xn(:, tr.testInd);
    Ytrue = Yn(:, tr.testInd);
    Yhat  = net(Xtest);

    dx = Yhat(1,:) - Ytrue(1,:);
    dy = Yhat(2,:) - Ytrue(2,:);
    err = sqrt(dx.^2 + dy.^2);

    rmse(i) = sqrt(mean(err.^2));
end

% RMSE vs. neuron count
figure;
plot(sizes, rmse, '-o');
xlabel('Hidden-layer neuron count');
ylabel('RMSE [m]');
