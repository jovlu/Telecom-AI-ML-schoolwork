clear; clc; close all;

pathToImages = "CNN_dataset/train";
imds = imageDatastore(pathToImages,"IncludeSubfolders",true,"LabelSource","foldernames");
[trainImgs,valImgs,testImgs] = splitEachLabel(imds,0.7,0.15,"randomized");

net = squeezenet;
inputSize = net.Layers(1).InputSize(1:2);
lgraph = layerGraph(net);
numClasses = numel(categories(imds.Labels));
lgraph = replaceLayer(lgraph,"conv10",convolution2dLayer(1,numClasses,"Name","new_conv","WeightLearnRateFactor",10,"BiasLearnRateFactor",10));
lgraph = replaceLayer(lgraph,"ClassificationLayer_predictions",classificationLayer("Name","new_classoutput"));

augmenter = imageDataAugmenter("RandXReflection",true);
trainds = augmentedImageDatastore([inputSize 3],trainImgs,"DataAugmentation",augmenter);
valds   = augmentedImageDatastore([inputSize 3],valImgs);
testds  = augmentedImageDatastore([inputSize 3],testImgs);

options = trainingOptions("adam","InitialLearnRate",1e-4,"MaxEpochs",8,"MiniBatchSize",16,"ValidationData",valds,"Plots","training-progress","Verbose",false);
netTransfer = trainNetwork(trainds,lgraph,options);

trainPreds = classify(netTransfer,trainds);
trainAcc = mean(trainPreds == trainImgs.Labels);
fprintf("Accuracy (TRAIN): %.2f%%\n",100*trainAcc);

[testPreds,testScores] = classify(netTransfer,testds);
testAcc = mean(testPreds == testImgs.Labels);
fprintf("Accuracy (TEST): %.2f%%\n",100*testAcc);
figure; confusionchart(testImgs.Labels,testPreds);
