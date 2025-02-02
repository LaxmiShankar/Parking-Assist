%-----------------AI DETECTION---------------------------------%
%Source: https://www.mathworks.com/help/vision/ug/object-detection-using-yolov4-deep-learning.html
%The code in this section is part of an example script for training
%YOLOV4Models
%--------------------------------------------------------------%
function data = preprocessData(data,targetSize)
for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    bboxes = data{ii,2};
    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    data(ii,1:2) = {I,bboxes};
end
end
function detector = downloadPretrainedYOLOv4Detector()
% Download a pretrained yolov4 detector.
if ~exist("yolov4TinyVehicleExample_24a.mat", "file")
    if ~exist("yolov4TinyVehicleExample_24a.zip", "file")
        disp("Downloading pretrained detector...");
        pretrainedURL = "https://ssd.mathworks.com/supportfiles/vision/data/yolov4TinyVehicleExample_24a.zip";
        websave("yolov4TinyVehicleExample_24a.zip", pretrainedURL);
    end
    unzip("yolov4TinyVehicleExample_24a.zip");
end
pretrained = load("yolov4TinyVehicleExample_24a.mat");
detector = pretrained.detector;
end
%Loading Datasets
unzip vehicleDatasetImages.zip
data = load("vehicleDatasetGroundTruth.mat");
vehicleDataset = data.vehicleDataset;
vehicleDataset(1:4,:)
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);
rng("default");
shuffledIndices = randperm(height(vehicleDataset));
%Selects 60% for training and 10% for validation
idx = floor(0.6 * length(shuffledIndices) );
trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);
validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);
testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);
imdsTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,"vehicle"));
imdsValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
bldsValidation = boxLabelDatastore(validationDataTbl(:,"vehicle"));
imdsTest = imageDatastore(testDataTbl{:,"imageFilename"});
bldsTest = boxLabelDatastore(testDataTbl(:,"vehicle"));
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
validateInputData(trainingData);
validateInputData(validationData);
validateInputData(testData);
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
reset(trainingData);
%object detector
inputSize = [416 416 3];
className = "vehicle";
rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 6;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)};
detector = yolov4ObjectDetector("tiny-yolov4-coco",className,anchorBoxes,InputSize=inputSize);

%Supporting functions
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
options = trainingOptions("adam", ...
    GradientDecayFactor=0.9, ...
    SquaredGradientDecayFactor=0.999, ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="none", ...
    MiniBatchSize=4, ...
    L2Regularization=0.0005, ...
    MaxEpochs=80, ...
    DispatchInBackground=true, ...
    ResetInputNormalization=true, ...
    Shuffle="every-epoch", ...
    VerboseFrequency=20, ...
    ValidationFrequency=1000, ...
    CheckpointPath=tempdir, ...
    ValidationData=validationData, ...
    OutputNetwork="best-validation-loss");
%train detector
doTraining = false;
if doTraining       
    % Train the YOLO v4 detector.
    [detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);
else
    % Load pretrained detector for the example.
    detector = downloadPretrainedYOLOv4Detector();
end

function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

%----------------------------END OF MODEL TRAINING------------------------------------------------------------%

%------------------------FINAL PROJECT SCRIPT-------%

%Load image of desired parking lot 
img = imread("C:\Users\laxmi\Downloads\Honors_Project_Code\Manual_ROI_Images\Parkinglotimages\lottest5.jpg");
imshow(img);
title('Draw rectangles around each parking slot and double click it when done');

% Manual ROI Selection
%Input data on your parking lot characteristics: number of rows, columns
%and total slots
num_rows = input('Enter the number of rows ');
num_cols= input('Enter the number of columns: ');
num_slots = input('Enter the number of parking slots to select: ');

file_names = cell(1, num_slots); % Cell array to store filenames
roi_positions = cell(1, num_slots);

for i = 1:num_slots
    disp('Draw rectangle for parking slot');
    h = drawrectangle(); % Draw rectangle ROI
    wait(h);  % Wait for the user to finish drawing the rectangle 
    position = h.Position;% Get the position of the rectangle
    roi_positions{i} = position; %updating it into a position index to save slot positions
end

imgtest = img; %ANALYZING STATIC IMAGES

%imgtest = imread("C:\Users\laxmi\Downloads\matlab parking
%images\stickermaker\lottest5.jpg"); % USE THIS LINE FOR VIDEO FOOTAGE 
for i = 1:num_slots
    disp('Draw rectangle for parking slot');
    htest = drawrectangle('Position',roi_positions{i});
    positiontest = htest.Position;
    
    % Extract the selected slot image
    slot_img = imcrop(imgtest, positiontest);

    % Save the selected slot image
    file_name = sprintf('manual_slot_%d.jpg', i);
    imwrite(slot_img, file_name);

    % Store the filename for later use
    file_names{i} = file_name;
end

disp('Manual selection completed. Slots saved as manual_slot_X.jpg.');

% Display the resulting images from saved files and the AI detection layer.
% Store if vehicle is detected in Slot_ID array. 

slot_ID = cell(1, num_slots); %initializing id array with zeros
for i = 1:num_slots
    slot_ID{i} = 0;
end

for i = 1:num_slots
    slot_img = imread(file_names{i});
    filename = file_names{i};
    I = imread(filename);
    [bboxes,scores,labels] = detect(detector,I);
    I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
    figure
    imshow(I)
    title(sprintf('Manual Slot and score %d', i))

    if scores > 0.2 %if AI detects a vehicle above 20% accuracy, set that slot's ID value to 1. 0 if not detected.
        slot_ID{i} = 1;
        fprintf('Slot %d is full.\n', i);
    else
        slot_ID{i} = 0;
        fprintf('Slot %d is empty.\n', i);
    end
end

%---------------------PARKING LOT GUI-------------------------------------%

slot_ID_GUI_unflipped = cell2mat(slot_ID);
slot_ID_GUI = cell2mat(slot_ID);
disp(slot_ID_GUI)
%convert cell array to regular array

scrsz = get(groot, 'ScreenSize');
MAIN = figure('Position', [100, 100, scrsz(3)-200, scrsz(4)-200], 'Color', [0.3, 0.3, 0.3], 'MenuBar', 'none', 'NumberTitle', 'off', 'Pointer', 'hand', 'Visible', 'on');

rect_width = 250; % Width of each parking space
rect_height = 500; % Height of each parking space
spacing = 10; % Spacing between rectangles
row_spacing = 0; % Spacing between rows

%sticker for visualizing occupied space
car_image = imread('car image.png'); 

% Define new width and height for the image
desired_width = rect_width * 0.9; 
desired_height = size(car_image, 1) * (desired_width / size(car_image, 2)); 

% Resize the image
car_image = imresize(car_image, [desired_height, desired_width]);

% Define slot_ID array indicating whether to display the car image
% Create an axes object for the entire figure
axes('Position', [0, 0, 1, 1], 'Visible', 'off');

%Creating a grid and initializing 
for r = 0:num_rows-1
    for c = 0:num_cols-1

       index = r*num_cols + c + 1; % Adjust for MATLAB's 1-based indexing 

        % Calculate position
        x = (rect_width + spacing) * c;
        y = (rect_height + row_spacing) * r;
        
        % Draw the rectangle
        rectangle('Position', [x, y, rect_width, rect_height], 'FaceColor', [0.3, 0.3, 0.3], 'EdgeColor', 'w', 'LineWidth', 2);
        
        % Displays image of a car in occupied slots based on AI detection
        if slot_ID_GUI(index) == 1
     
            img_x = x + (rect_width - desired_width) / 2;
            img_y = y + (rect_height - desired_height) / 2;
            
            % Place the image within the grid cell
            image('CData', car_image, 'XData', [img_x , img_x + desired_width], 'YData', [img_y, img_y + desired_height]);
        else
           
            %Displays an OPEN! message in unoccupied slots
            circle_radius = min(rect_width, rect_height) / 2.5; 
            circle_x = x + rect_width / 2;
            circle_y = y + rect_height / 2;

            % Draw a green circle
            rectangle('Position', [circle_x - circle_radius, circle_y - circle_radius, 2 * circle_radius, 2 * circle_radius], ...
                      'Curvature', [1, 1], 'FaceColor', 'g', 'EdgeColor', 'none');
            
            % Display text for empty slot
            text('Position', [circle_x, circle_y], 'String', 'OPEN!', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
end

% Set the aspect ratio and turn off axis
axis equal;
axis off;
