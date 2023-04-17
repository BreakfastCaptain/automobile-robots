% Load the pre-trained network
pretrainedNetwork = fullfile('deeplabv3plusResnet18CamVid.mat');  
data = load(pretrainedNetwork);
net = data.net;

% Define the required classes
classes = [
    "Vehicles"
    "Bicycles and Motorcycles"
    "Pedestrians"
    "Drivable Surface"
    "Others"
    ];

% Read the image containing the classes used to train the network for classification
% Change different pictures to get different results
I = imread('photo2_imp.jpg');

% Resize the image to the input size of the network
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));

% Perform semantic segmentation using semanticseg function and pre-trained networks
C = semanticseg(I,net);

% Use labeloverlay function to overlay the segmentation result on the image. 
cmap = camvidColorMap;
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.5);
figure

% Show the image
imshow(B)

% Define labels with the same color as the classification results 
cmap1 = [
    0 0 255      % Vehicles
    0 255 0     % Bicycles and motorcycles
    255 255 0       % Pedestrians
    255 0 0    % Drivable surface
    0 0 0   % Others
    ];
cmap1 = cmap1 ./ 255;

% Plot the label legend on the right side of the image
pixelLabelColorbar(cmap1,classes);

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis, 
% gca returns the current coordinate area in the current chart window
colormap(gca,cmap)

% Add colorbar to current figure
c = colorbar('peer', gca);

% Use class names for tick marks, 
% since the pre-training network is 11 classifications, the number of classes needs to be changed to 5
c.TickLabels = classNames;
numClasses = 5;

% Center tick labels
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end

function cmap = camvidColorMap()
% Define the colormap used by CamVid dataset.
% Change the remaining irrelevant recognition objects to the same color,
% and reduce the subsequent classification from 11 categories to 5 categories
cmap = [
    0 0 0   % Sky
    0 0 0       % Building
    0 0 0   % Pole
    255 0 0    % Road
    0 0 0     % Pavement
    0 0 0     % Tree
    0 0 0   % SignSymbol
    0 0 0    % Fence
    0 0 255      % Car
    255 255 0       % Pedestrian
    0 255 0     % Bicyclist
    ];

% Normalize between [0 1], because the colormap function must enter unitized data
cmap = cmap ./ 255;
end