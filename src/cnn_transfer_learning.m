clear; clc; close all;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
datasetRoot = fullfile(projectRoot, 'data', 'CNN_dataset');
trainDir = fullfile(datasetRoot, 'train');
valDir = fullfile(datasetRoot, 'valid');
testDir = fullfile(datasetRoot, 'test');
datasetZip = fullfile(projectRoot, 'data', 'CNN_dataset.zip');

if ~isfolder(trainDir) && isfile(datasetZip)
    unzip(datasetZip, fullfile(projectRoot, 'data'));
end

if ~isfolder(trainDir) || ~isfolder(valDir) || ~isfolder(testDir)
    error('Missing CNN dataset split folders. Expected train, valid, and test under: %s', datasetRoot);
end

trainImgs = imageDatastore(trainDir,"IncludeSubfolders",true,"LabelSource","foldernames");
valImgs = imageDatastore(valDir,"IncludeSubfolders",true,"LabelSource","foldernames");
testImgs = imageDatastore(testDir,"IncludeSubfolders",true,"LabelSource","foldernames");

net = squeezenet;
inputSize = net.Layers(1).InputSize(1:2);
lgraph = layerGraph(net);
numClasses = numel(categories(trainImgs.Labels));
lgraph = replaceLayer(lgraph, "conv10", convolution2dLayer(1, numClasses, ...
    "Name", "new_conv", ...
    "WeightLearnRateFactor", 10, ...
    "BiasLearnRateFactor", 10));
lgraph = replaceLayer(lgraph, "ClassificationLayer_predictions", ...
    classificationLayer("Name", "new_classoutput"));

augmenter = imageDataAugmenter("RandXReflection",true);
trainds = augmentedImageDatastore([inputSize 3], trainImgs, "DataAugmentation", augmenter);
valds   = augmentedImageDatastore([inputSize 3], valImgs);
testds  = augmentedImageDatastore([inputSize 3], testImgs);

options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-4, ...
    "MaxEpochs", 8, ...
    "MiniBatchSize", 16, ...
    "ValidationData", valds, ...
    "Plots", "training-progress", ...
    "Verbose", false);
netTransfer = trainNetwork(trainds, lgraph, options);

trainPreds = classify(netTransfer, trainds);
trainAcc = mean(trainPreds == trainImgs.Labels);
fprintf("Accuracy (TRAIN): %.2f%%\n",100*trainAcc);

testPreds = classify(netTransfer, testds);
testAcc = mean(testPreds == testImgs.Labels);
fprintf("Accuracy (TEST): %.2f%%\n",100*testAcc);
figure; confusionchart(testImgs.Labels, testPreds);
