clear all, close all;

%UCITAJ
T = readtable('RSSI Database.xls', 'VariableNamingRule', 'preserve');

%PREBACI SVE U NUMERICKI OBLIK (AP VAN DOMETA STAVITI NA -120dBm)
M = zeros(height(T), width(T));
for i = 1:width(T)
    col = T.(i);
    if iscell(col) || isstring(col)
        col = string(col);
        col( col=="*") = "-120";
        col = str2double(col);
    end
    col(isnan(col)) = -120;
    M(:,i) = col;
end

%UZETI PROSEK SVIH ORIJENTACIJA
data = M(:,1:end-2);
xy_all = M(:,end-1:end);

nLoc = size(M,1)/4;
Xavg  = zeros(nLoc, size(data,2));
XYavg = zeros(nLoc, 2);

for g = 1:nLoc
    idx = (g-1)*4 + (1:4);
    Xavg(g,:)  = mean(data(idx,:), 1);
    XYavg(g,:) = mean(xy_all(idx,:), 1);
end

xy  = XYavg;
mat = [Xavg xy];



%PRIKAZ MAPE
x_m = xy(:,1) * 0.1;
y_m = xy(:,2) * 0.1;

figure(1);
scatter(x_m, y_m, '.');
xlabel('x [m]');
ylabel('y [m]');
axis equal



%PODELA U TEST I TRAIN MODEL
N = size(mat,1);
idx = randperm(N);

Ntr = ceil(0.9*N);
tr_idx = idx(1:Ntr);
te_idx = idx(Ntr+1:end);

trainTbl = array2table(mat(tr_idx,:));
testTbl  = array2table(mat(te_idx,:));

% PRIKAZ TRAIN I TEST SCATTER
x_tr = trainTbl{:,end-1} * 0.1;
y_tr = trainTbl{:,end}   * 0.1;

x_te = testTbl{:,end-1} * 0.1;
y_te = testTbl{:,end}   * 0.1;

figure(2);
scatter(x_tr, y_tr, 'blue'); % plava - train
hold on;
scatter(x_te, y_te, 'red'); % crvena - test
xlabel('x [m]');
ylabel('y [m]');
legend("Train", "Test");
axis equal;

%KNN

%PODELA
Xtr = trainTbl{:,1:end-2};
Xte = testTbl{:,1:end-2};
Ytr = trainTbl{:,end-1:end};
Yte = testTbl{:,end-1:end};

%Z NORMALIZACIJA
mu  = mean(Xtr,1);
sig = std(Xtr,0,1);
sig(sig==0) = 1;
Xtr = (Xtr - mu) ./ sig;
Xte = (Xte - mu) ./ sig;

%NADJI K NAJSLICNIJIH PO RSSI
k = 3;
[idx, ~] = knnsearch(Xtr, Xte, 'K', k, 'Distance', 'euclidean');

%UZMI PROSEK LOKACIJA NAJSLICNIJIH PO RSSI
Nte = size(Xte,1);
Yhat = zeros(Nte,2);

for i = 1:Nte
    nn = idx(i,:);
    Yhat(i,:) = mean(Ytr(nn,:), 1);
end

%GRESKE
err_m = sqrt(sum(((Yhat - Yte) * 0.1).^2, 2));
RMSE = sqrt(mean(err_m.^2));
MEAN = mean(err_m);
STD  = std(err_m);
fprintf('k=%d | RMSE=%.3f m | MEAN=%.3f m | STD=%.3f m\n', k, RMSE, MEAN, STD);

%HISTOGRAM GRESKE
figure; histogram(err_m,30); grid on;
xlabel('Greska u m'); ylabel('Broj tacaka');

figure; hold on;
scatter(Yte(:,1)*0.1,  Yte(:,2)*0.1,  'bo');
scatter(Yhat(:,1)*0.1, Yhat(:,2)*0.1, 'rx');
xlabel('x [m]'); ylabel('y [m]'); legend('Tacna','Pretpostavljena');
axis equal;


%K SWEEP
Ks = 1:10;
rmse = zeros(size(Ks));

for ii = 1:numel(Ks)
    k = Ks(ii);
    idx = knnsearch(Xtr, Xte, 'K', k, 'Distance', 'euclidean');
    Yhat = zeros(size(Yte));
    for i = 1:size(Xte,1)
        Yhat(i,:) = mean(Ytr(idx(i,:),:), 1);
    end
    e = sqrt(sum(((Yhat - Yte) * 0.1).^2, 2));
    rmse(ii) = sqrt(mean(e.^2));
end

figure; plot(Ks, rmse, '-o'); grid on;
xlabel('k'); ylabel('RMSE');
