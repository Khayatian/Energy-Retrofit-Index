% Note:
% Define the inputs X (samples in rows, features as their columns, e.g. [5000,5])
% Define the target T (targets in rows, e.g. [5000,1])

% Transpose the data
xTrain=X';
tTrain=T';

% for reproducibility (remove if you want different results when...
%                       ... re-running the same script multiple times)
rng('default')

% Size of the first autoencoder (over representing the inputs xTrain)
hiddenSize1 = 8;

%Training the first Autoencoder
autoenc1 = trainAutoencoder(xTrain,hiddenSize1, ...
    'MaxEpochs',100, ...
    'EncoderTransferFunction','logsig',...
    'DecoderTransferFunction','purline',...
    'L2WeightRegularization',0.001, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.0001, ...
    'TrainingAlgorithm','trainscg',...
    'ScaleData',true);

% Encode the new feature with dimensionality of 8
feat1 = encode(autoenc1,xTrain);

% Size of the second Autoencoder (sparsing the over-represented features)
hiddenSize2 = 5;

% Training the second Autoencoder
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',50, ...
    'EncoderTransferFunction','satlin',...
    'DecoderTransferFunction','purelin',...
    'L2WeightRegularization',0.001, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.01, ...
    'TrainingAlgorithm','trainscg',...
    'ScaleData',false);

% Encode the  new new feature with dimensionality of 5
feat2 = encode(autoenc2,feat1);

% Create a Network with 10 logistic nodes, one hidden layer
net = feedforwardnet(10);

% Train the vanilla Neural Net, 5 inputs (from Autoencoder 2) and one target (EPI)
Vnet = trainlm(net,feat2,tTrain);

% Stack Autoencoder1, Autoencoder2, vanilla Neural Net to create Deep Net
deepnet = stack(autoenc1,autoenc2,Vnet);

% Train the Deep Net
deepnet = train(deepnet,xTrain,tTrain);

% Extract the outputs of the trained model. (Dont worry if the performance
% is low, as our mission is not prediction)
y = deepnet(xTrain);

% Create a copy of Deep Net and call it Deep Net2
deepnet2=deepnet;

% Remove the link between layers 4 and 5
deepnet2.layerConnect(5,4) = 0;

% Create a copy of Deep Net2 and call it Deep Net3
deepnet3=deepnet2;

% Connect the logistic layer (L4) with the output node
deepnet3.outputConnect(1,4) = 1;

% Remove the link between linear layer (L5) and output node
deepnet3.outputConnect(1,5) = 0;

% Extract the outputs of the truncated Deep Net3
y2=deepnet3(xTrain);

% Size of the 3rd Autoencoder, I call it the "ten2one"
hiddenSize3 = 1;

% Use the outputs of Deep Net3 and feed into Autoencoder3
autoenc3 = trainAutoencoder(y2,hiddenSize3, ...
    'MaxEpochs',10000, ...
    'L2WeightRegularization',0, ...
    'SparsityRegularization',0, ...
    'SparsityProportion',0, ...
    'ScaleData', true);

% Encode the  new new feature with dimensionality of 1
feat3 = encode(autoenc3,y2);

% Transpose the data to extract ERI index
ERI=feat3';