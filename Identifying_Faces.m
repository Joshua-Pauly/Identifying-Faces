% Lesson 9 HW Imag classification
load('face_pos.mat');
positiveInstances = struct('imageFilename',gTruth.DataSource.Source,...
'objectBoundingBoxes',table2cell(gTruth.LabelData));

addpath(fullfile(pwd,'positives'));
addpath(fullfile(pwd,'negatives'));

negativeFolder = fullfile(pwd,'negatives');
 
trainCascadeObjectDetector('facesModel.xml',positiveInstances,negativeFolder,FalseAlarmRate=0.1,NumCascadeStages=5);

faceDetector = vision.CascadeObjectDetector('facesModel.xml');
faceDetector.MergeThreshold = 27;

%Nature
img = imread('nature_image.jpg');
bbox = step(faceDetector,img);
detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'');
 
figure;
imshow(detectedImg);

%Faces
faces = imread('face_test.jpg');
bbox = step(faceDetector,faces);
detectedImg = insertObjectAnnotation(faces,'rectangle',bbox,'');
 
figure;
imshow(detectedImg);


% train ACF
load('face_pos.mat');
facesGTruth = selectLabels(gTruth,"face");
addpath(fullfile('positive'));
trainingData = objectDetectorTrainingData(facesGTruth);
detector = trainACFObjectDetector(trainingData,"NumStages",3);
save('faceACFDetector.mat','detector');
load('faceACFDetector.mat');

% test ACF

[bboxes, scores] = detect(detector,img, "Threshold",.3);
idx = scores >= 40;
boxes1 = bboxes(idx,:);
scores1 = scores(idx);
[boxes1,scores1] = selectStrongestBbox(boxes1,scores1,'OverlapThreshold',.1);
detectedImg = insertObjectAnnotation(img,'rectangle',boxes1,'');
imshow(detectedImg);
 
[bboxes, scores] = detect(detector,faces, "Threshold",.3);
idx = scores >= 40;
boxes2 = bboxes(idx,:);
scores2 = scores(idx);
[boxes2,scores2] = selectStrongestBbox(boxes2,scores2,'OverlapThreshold',.1);
detectedImg = insertObjectAnnotation(faces,'rectangle',boxes2,'');
imshow(detectedImg);