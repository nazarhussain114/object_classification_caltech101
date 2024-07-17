% % %https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html
clear 
clc
close all
% % end
% %% Dataset Paths
outputFolder='D:\Caltech101';
rootFolder = fullfile(outputFolder, '50_ObjectCategories');

allfoldernames= struct2table(dir(rootFolder));
for (i=3:height(allfoldernames))
    new(i-2)=allfoldernames.name(i);
end
clear i
categories=new;
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource','foldernames');
tbl = countEachLabel(imds);
 minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% % Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% % Notice that each set now has exactly the same number of images.
countEachLabel(imds);
% % Find the first instance of an image for each category
% %% Pretrained Net AlexNet
% % net = alexnet();
 net = inceptionv3();
% % net = alexnet();
 net.Layers(1);
 net.Layers(end);
% 
imr=net.Layers(1, 1).InputSize(:,1);
imc=net.Layers(1, 1).InputSize(:,2);
% 
 imds.ReadFcn = @(filename)readAndPreprocessImage(filename,imr,imc);
 [trainingSet, testSet] = splitEachLabel(imds, 0.5, 'random');
% % Get the network weights for the second convolutional layer
% % w1 = net.Layers(2).Weights;
% %%   Resize weigts for vgg only
% % w1 = imresize(w1,[imr imc]);
% %%
featureLayer = 'avg_pool';
tic
% %%
 trainingFeatures = activations(net, trainingSet, featureLayer, ...
 'MiniBatchSize', 10, 'OutputAs', 'columns');
% 
% %%
% % Get training labels from the trainingSet
trainingLabels = cellstr(trainingSet.Labels);
% 
% % Train multiclass SVM classifier using a fast linear solver, and set
% % 'ObservationsIn' to 'columns' to match the arrangement used for training
% % features.
 tic
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% 
trainingFeatures =trainingFeatures';
features=horzcat(trainingLabels,num2cell(trainingFeatures));
 features=cell2table(features);
% 
save('network_new_train_inceptionv3_','trainingFeatures','trainingLabels', 'classifier');












































% % %https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html
% clc;
% close all;
% clear 
% outputFolder='D:\datasets\Caltech101\';
% rootFolder = fullfile(outputFolder, '100_ObjectCategories');
% allfoldernames= struct2table(dir(rootFolder));
% for (i=3:height(allfoldernames))
%     new(i-2)=allfoldernames.name(i);
% end
% clear i
% % categories = {'airplanes', 'ferry', 'laptop'};
% categories=new;
% imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
% tbl = countEachLabel(imds);
% minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% % Use splitEachLabel method to trim the set.
% imds = splitEachLabel(imds, minSetCount, 'randomize');
% % Notice that each set now has exactly the same number of images.
% countEachLabel(imds);
% minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% % Use splitEachLabel method to trim the set.
% imds = splitEachLabel(imds, minSetCount, 'randomize');
% % Notice that each set now has exactly the same number of images.
% countEachLabel(imds);
% % Find the first instance of an image for each category
% 
% %% Pretrained Net AlexNet
% net = inceptionv3();
% net.Layers(1);
% net.Layers(end);
% numel(net.Layers(end).ClassNames);
% 
% 
% imr=net.Layers(1, 1).InputSize(:,1);
% imc=net.Layers(1, 1).InputSize(:,2);
% 
% imds.ReadFcn = @(filename)readAndPreprocessImage(filename,imr,imc);
% [trainingSet, testSet] = splitEachLabel(imds, 0.5, 'randomize');
% % Get the network weights for the second convolutional layer
% % w1 = net.Layers(2).Weights;
% % Scale and resize the weights for visualization
% % w1 = mat2gray(w1);
% %  w1 = imresize(w1,[228 228]);
% % Display a montage of network weights. There are 96 individual sets of
% % weights in the first layer.
% % figure
% % montage(w1);
% % title('First convolutional layer weights');
% featureLayer = 'avg_pool';
% 
% trainingFeatures = activations(net, trainingSet, featureLayer, ...
%     'MiniBatchSize', 10, 'OutputAs', 'columns');
% % trainingFeatures = activations(net, trainingSet, featureLayer);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% % Get training labels from the trainingSet
% trainingLabels = trainingSet.Labels;
% 
% % Train multiclass SVM classifier using a fast linear solver, and set
% % 'ObservationsIn' to 'columns' to match the arrangement used for training
% % features.
% classifier = fitcecoc(trainingFeatures, trainingLabels, ...
%     'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% %%
% %%
% % Extract test features using the CNN
% testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32, 'OutputAs', 'columns');
% % Pass CNN image features to trained classifier
% predictedLabels = predict(classifier, testFeatures);
% % Get the known labels
% testLabels = testSet.Labels;
% %%
% % x feature vector
% x=trainingFeatures';
% % y is label
% y=cellstr(trainingLabels);
% %   XY is feature,label
% xy=array2table(x);
% xy.type=y;
% save('xy_inception','trainingFeatures','classifier','net');













