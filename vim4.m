clc; clear;

% Load data
T = readtable('RSSI Database.xls','VariableNamingRule','preserve');
apCols = 1:8; xyCols = 9:10;
noiseFloor = -120; metersPerPixel = 0.1;

% Convert APs
AP = zeros(height(T), numel(apCols));
for j=1:numel(apCols)
    col = T{:,apCols(j)};
    if iscell(col) || isstring(col) || ischar(col)
        col = string(col);
        col(col=="*") = string(noiseFloor);
        col = str2double(col);
    end
    col(isnan(col)) = noiseFloor;
    AP(:,j) = col;
end

% XY
XY = T{:,xyCols};
if iscell(XY) || isstring(XY) || ischar(XY)
    XY = str2double(string(XY));
end

% ==== Repeat 100 trials and keep the best (smallest RMSE) ====
bestRMSE = inf;
bestAcc = NaN; bestSt = NaN; bestMe = NaN;
bestErrs_m = []; bestTrial = 0;

threshold = 10;  % meters

for trial = 1:100
    % Split 80/20
    N = size(AP,1); idx = randperm(N);
    nTr = round(0.8*N);
    tr = idx(1:nTr); te = idx(nTr+1:end);

    Xtr = AP(tr,:); yXtr = XY(tr,1); yYtr = XY(tr,2);
    Xte = AP(te,:); yXte = XY(te,1); yYte = XY(te,2);

    % Train SVMs (your settings)
    mdlX = fitrsvm(Xtr,yXtr,'KernelFunction','polynomial','Standardize',true,'KernelScale','auto');
    mdlY = fitrsvm(Xtr,yYtr,'KernelFunction','polynomial','Standardize',true,'KernelScale','auto');

    % Predict
    yXhat = predict(mdlX,Xte);
    yYhat = predict(mdlY,Xte);

    % Errors
    errPix = hypot(yXhat-yXte, yYhat-yYte);
    errs_m = errPix * metersPerPixel;   % convert to meters
    rmse_m = sqrt(mean(errPix.^2)) * metersPerPixel;
    acc = mean(errs_m < threshold) * 100;
    st = std(errs_m);
    me = mean(errs_m);

    % Track best
    if rmse_m < bestRMSE
        bestRMSE = rmse_m;
        bestAcc  = acc;
        bestSt   = st;
        bestMe   = me;
        bestErrs_m = errs_m;
        bestTrial = trial;
    end
end

% ==== Display best run metrics exactly like your prints ====
fprintf('Best over 100 trials (trial #%d)\n', bestTrial);
fprintf('RMSE = %.2f m\n', bestRMSE);
fprintf('Accuracy (<= %.1f m) = %.2f%%\n', threshold, bestAcc);
fprintf('std = %.2f m\n', bestSt);
fprintf('MEAN = %.2f m\n', bestMe);

% --- Histogram of errors for the best trial ---
figure;
histogram(bestErrs_m, 20);   % 20 bins
xlabel('Localization error [m]');
ylabel('Number of test samples');
title(sprintf('Distribution of localization errors (best trial #%d)', bestTrial));
grid on;
