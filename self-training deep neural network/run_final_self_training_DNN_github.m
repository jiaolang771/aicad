% Author:- Redha Ali
% Cincinnati Children’s Hospital Medical Center (CCHMC)
% Date:- 03/01/2021

%% Clear screen and workspace
close all; clear; clc;

%% Load data

% Create labeled random dataset
Labeled_dataset = rand(103,4005);
truth_labels = randi(2, 103, 1)';

% Create unlabeled random dataset
unlabeled_dataset = rand(240,4005);

%% Self-training DNN approach

% define number of neurons for each layer
nodes_1 = 1100;
nodes_2 = 35;
nodes_3 = 35;

% Create 5 K-fold cross-validation partition for data.
K_folds = 5;
CVO = cvpartition(truth_labels,'KFold',K_folds,'Stratify',true);

for idx = 1:K_folds
    
    fprintf('Processing %d of %d folds \n\n',idx,K_folds )
    
    % Training set indices for a cross-validation partition
    train_index = CVO.training(idx);
    
    % Test set indices for a cross-validation partition.
    test_index = CVO.test(idx);
    
    % Test images for K fold
    TestingSet = (Labeled_dataset(test_index,:));
    
    % Train images for k fold
    TraningSet = (Labeled_dataset(train_index,:));
    TraningLabels = categorical(truth_labels(train_index));
    
    % Random partition for hold-out validation
    CVO1 = cvpartition(TraningLabels,'HoldOut',0.2,'Stratify',true);
    
    % Training and validation indices
    training_index = gather(CVO1.training);
    Vald_index =   gather(CVO1.test);
    
    
    Valid_set = TraningSet(Vald_index,:);
    TrainSet = TraningSet(training_index,:);
    Train_label = categorical(TraningLabels(training_index));
    Valid_label =  categorical(TraningLabels(Vald_index));
    
    % Invese Frequency Class Weights for weighted loss layer
    totalNumberOfsample = length(Train_label);
    frequency = countcats(categorical(Train_label)) / totalNumberOfsample;
    invFreqClassWeights = 1./frequency;
    
    % Create DNN
    layers = [
        featureInputLayer(4005,"Name","featureinput","Normalization","rescale-zero-one")
        batchNormalizationLayer("Name","BN1")
        reluLayer("Name","relu1")
        fullyConnectedLayer(1200,"Name","fc1")
        batchNormalizationLayer("Name","BN2")
        fullyConnectedLayer(2,"Name","fc2")
        softmaxLayer("Name","softmax")
        weightedClassificationLayer(invFreqClassWeights)
        ];
    
    % Options for training a neural network
    options = trainingOptions('rmsprop', ...
        'InitialLearnRate',1e-03,...
        'MaxEpochs',50, ...
        'MiniBatchSize',16, ...
        'ValidationData',{Valid_set,Valid_label},...
        'ValidationFrequency',20,...
        'ExecutionEnvironment','gpu',...
        'Shuffle','every-epoch',...
        'Verbose', false,...
        'ValidationFrequency',20);
    
    % Train
    Teacher_net = trainNetwork(TrainSet,Train_label,layers,options);
    close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
    
    % Test
    [Teacher_Pred,Teacher_S] = classify(Teacher_net,TestingSet);
    
    % Produce a pseudo label for all 240 samples using teacher network.
    [pseudolabels_1iter,pseudoScore] = classify(Teacher_net,unlabeled_dataset,'ExecutionEnvironment','gpu');
    
    % Combine both dataset truth and pseudo ,pseudoScore
    Combine_trainset = [TrainSet;unlabeled_dataset];
    Combine_trainset_labels = [Train_label';pseudolabels_1iter];
    
    % Invese Frequency Class Weights for weighted loss layer
    totalNumberOfsample2 = length(Combine_trainset_labels);
    frequency2 = countcats(categorical(Combine_trainset_labels)) / totalNumberOfsample2;
    invFreqClassWeights2 = 1./frequency2;
    
    % Create DNN
    Studen_layers = [
        featureInputLayer(4005,"Name","featureinput","Normalization","rescale-zero-one")
        batchNormalizationLayer("Name","BN1")
        softplusLayer("Name","softplus1")
        fullyConnectedLayer(nodes_1,"Name","fc1")
        batchNormalizationLayer("Name","BN2")
        softplusLayer("Name","softplus2")
        fullyConnectedLayer(nodes_2,"Name","fc3")
        batchNormalizationLayer("Name","BN3")
        softplusLayer("Name","softplus3")
        fullyConnectedLayer(nodes_3,"Name","fc4")
        batchNormalizationLayer("Name","BN4")
        softplusLayer("Name","softplus4")
        fullyConnectedLayer(2,"Name","fc_n")
        softmaxLayer("Name","softmax")
        weightedClassificationLayer(invFreqClassWeights2')
        ];
    
    % Options for training a neural network
    options2 = trainingOptions('rmsprop', ...
        'InitialLearnRate',1e-03,...
        'MaxEpochs',50, ...
        'MiniBatchSize',16, ...
        'ValidationData',{Valid_set,Valid_label},...
        'ValidationFrequency',20,...
        'ExecutionEnvironment','gpu',...
        'Shuffle','every-epoch',...
        'Verbose', false,...
        'ValidationFrequency',20);
    
    %Train
    Student_modelI = trainNetwork(Combine_trainset,Combine_trainset_labels,Studen_layers,options2);
    
    [StudentI_Pred,StudentI_S] = classify(Student_modelI,TestingSet);
    
    %Feature ranking
    multiplication_Weights = DNN_Featureranking(Student_modelI);
    Total_Weights(:,:,idx) = multiplication_Weights;
    
    % Produce a pseudo label for all 240 samples using teacher network.
    [pseudolabels_1iterII,pseudoScoreI] = classify(Student_modelI,unlabeled_dataset);
    
    % Combine both dataset truth and pseudo
    Combine_trainsetII = [TrainSet;unlabeled_dataset];
    Combine_trainset_labelsII = [Train_label';pseudolabels_1iterII];
    
    % Invese Frequency Class Weights for weighted loss layer
    totalNumberOfsample3 = length(Combine_trainset_labelsII);
    frequency3 = countcats(categorical(Combine_trainset_labelsII)) / totalNumberOfsample3;
    invFreqClassWeights3 = 1./frequency3;
    
    % Create DNN
    Studen_layersII = [
        featureInputLayer(4005,"Name","featureinput","Normalization","rescale-zero-one")
        batchNormalizationLayer("Name","BN1")
        softplusLayer("Name","softplus1")
        fullyConnectedLayer(nodes_1,"Name","fc1")
        batchNormalizationLayer("Name","BN2")
        softplusLayer("Name","softplus2")
        fullyConnectedLayer(nodes_2,"Name","fc3")
        batchNormalizationLayer("Name","BN3")
        softplusLayer("Name","softplus3")
        fullyConnectedLayer(nodes_3,"Name","fc4")
        batchNormalizationLayer("Name","BN4")
        softplusLayer("Name","softplus4")
        fullyConnectedLayer(2,"Name","fc_n")
        softmaxLayer("Name","softmax")
        weightedClassificationLayer(invFreqClassWeights3')
        ];
    
    % Options for training a neural network
    options3 = trainingOptions('rmsprop', ...
        'InitialLearnRate',1e-03,...
        'MaxEpochs',50, ...
        'MiniBatchSize',16, ...
        'ValidationData',{Valid_set,Valid_label},...
        'ValidationFrequency',20,...
        'ExecutionEnvironment','gpu',...
        'Shuffle','every-epoch',...
        'Verbose', false,...
        'ValidationFrequency',20);
    %Train
    Student_model_II = trainNetwork(Combine_trainsetII,Combine_trainset_labelsII,Studen_layersII,options3);
    [StudentII_Pred,StudentII_S] = classify(Student_model_II,TestingSet);
    
    % Produce a pseudo label for all 240 samples using teacher network.
    [pseudolabels_1iterIII,pseudoScoreIII] = classify(Student_model_II,unlabeled_dataset);
    
    % Combine both dataset truth and pseudo
    Combine_trainsetIII = [TrainSet;unlabeled_dataset];
    Combine_trainset_labelsIII = [Train_label';pseudolabels_1iterIII];
    
    % Invese Frequency Class Weights for weighted loss layer
    totalNumberOfsample4 = length(Combine_trainset_labelsIII);
    frequency4 = countcats(categorical(Combine_trainset_labelsIII)) / totalNumberOfsample4;
    invFreqClassWeights4 = 1./frequency4;
    
    % Create DNN
    Studen_layersIII = [
        featureInputLayer(4005,"Name","featureinput","Normalization","rescale-zero-one")
        batchNormalizationLayer("Name","BN1")
        softplusLayer("Name","softplus1")
        fullyConnectedLayer(nodes_1,"Name","fc1")
        batchNormalizationLayer("Name","BN2")
        softplusLayer("Name","softplus2")
        fullyConnectedLayer(nodes_2,"Name","fc3")
        batchNormalizationLayer("Name","BN3")
        softplusLayer("Name","softplus3")
        fullyConnectedLayer(nodes_3,"Name","fc4")
        batchNormalizationLayer("Name","BN4")
        softplusLayer("Name","softplus4")
        fullyConnectedLayer(2,"Name","fc_n")
        softmaxLayer("Name","softmax")
        weightedClassificationLayer(invFreqClassWeights4')
        ];
    
    % Options for training a neural network
    options4 = trainingOptions('rmsprop', ...
       'InitialLearnRate',1e-03,...
        'MaxEpochs',50, ...
        'MiniBatchSize',16, ...
        'ValidationData',{Valid_set,Valid_label},...
        'ValidationFrequency',20,...
        'ExecutionEnvironment','gpu',...
        'Shuffle','every-epoch',...
        'Verbose', false,...
        'ValidationFrequency',20);
    
    %Train
    Student_model_III = trainNetwork(Combine_trainsetIII,Combine_trainset_labelsII,Studen_layersIII,options4);
    
    [StudentIII_Pred,StudentIII_S] = classify(Student_model_III,TestingSet);
    
    % Produce a pseudo label for all 240 samples using teacher network.
    [pseudolabels_1iterIIII,pseudoScoreIIII] = classify(Student_model_III,unlabeled_dataset);

    %Combine both dataset truth and pseudo
    Combine_trainsetIIII = [TrainSet;unlabeled_dataset];
    Combine_trainset_labelsIIII = [Train_label';pseudolabels_1iterIIII];
    
    % Invese Frequency Class Weights for weighted loss layer
    totalNumberOfsample5 = length(Combine_trainset_labelsIIII);
    frequency5 = countcats(categorical(Combine_trainset_labelsIIII)) / totalNumberOfsample5;
    invFreqClassWeights5 = 1./frequency5;
    
    
    Studen_layers_IIII = [
        featureInputLayer(4005,"Name","featureinput","Normalization","rescale-zero-one")
        batchNormalizationLayer("Name","BN1")
        softplusLayer("Name","softplus1")
        fullyConnectedLayer(nodes_1,"Name","fc1")
        batchNormalizationLayer("Name","BN2")
        softplusLayer("Name","softplus2")
        fullyConnectedLayer(nodes_2,"Name","fc3")
        batchNormalizationLayer("Name","BN3")
        softplusLayer("Name","softplus3")
        fullyConnectedLayer(nodes_3,"Name","fc4")
        batchNormalizationLayer("Name","BN4")
        softplusLayer("Name","softplus4")
        fullyConnectedLayer(2,"Name","fc_n")
        softmaxLayer("Name","softmax")
        weightedClassificationLayer(invFreqClassWeights5')
        ];
    
    % Options for training a neural network
    options5 = trainingOptions('rmsprop',...
       'InitialLearnRate',1e-03,...
        'MaxEpochs',50, ...
        'MiniBatchSize',16, ...
        'ValidationData',{Valid_set,Valid_label},...
        'ValidationFrequency',20,...
        'ExecutionEnvironment','gpu',...
        'Shuffle','every-epoch',...
        'Verbose', false,...
        'ValidationFrequency',20);
    %Train
    Student_model_IIII = trainNetwork(Combine_trainsetIIII,Combine_trainset_labelsIIII,Studen_layers_IIII,options5);
    
    [StudentIIII_Pred,StudentIIII_S] = classify(Student_model_IIII,TestingSet);
    
    
    %predictions Labels and scores
    Teacher_pred_Labels(test_index) = Teacher_Pred;
    Teacher_pred_score(test_index,:) = Teacher_S;
    
    StudentI_pred_Labels(test_index) = StudentI_Pred;
    StudentI_pred_score(test_index,:) = StudentI_S;
    
    
    StudentII_pred_Labels(test_index) = StudentII_Pred;
    StudentII_pred_score(test_index,:) = StudentII_S;
    
    StudentIII_pred_Labels(test_index) = StudentIII_Pred;
    StudentIII_pred_score(test_index,:) = StudentIII_S;
    
    StudentIIII_pred_Labels(test_index) = StudentIIII_Pred;
    StudentIIII_pred_score(test_index,:) = StudentIIII_S;
    
    
end


