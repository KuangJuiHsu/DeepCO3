clearvars
close all
clc

%% Library
LibDir = [pwd '/Lib'];
addpath(LibDir)
addpath([LibDir '/matconvnet-1.0-beta24/'])
addpath([LibDir '/matconvnet-1.0-beta24/mex'])
addpath([LibDir '/matconvnet-1.0-beta24/simplenn'])
addpath([LibDir '/Solver'])
addpath([LibDir '/InstCosegEval'])
addpath([LibDir '/export_fig-master'])

%% Path
DatasetNameList = {'COCO_VOC', 'COCO_NONVOC', 'VOC12', 'SOC'};
DatasetDir = [pwd '/Dataset/'];
ImageDir = strcat(DatasetDir, DatasetNameList, '/Image/' ); % inputs
MaskDir = strcat(DatasetDir, DatasetNameList, '/Mask/' ); % ground truth
SalDir = strcat(DatasetDir, DatasetNameList, '/SalRes_ZhangICCV17/'); % saliency prior
RootDir = [pwd '/Result/']; % save the results
FinalResultSaveDir = strcat([pwd '/DeepInstCosegResult/'], DatasetNameList);
GPUID = 1; % gpu id

%% Parameters
SelectionParams = [0.8 1e-5]; % proposal selection 
learningRate = 0.000001 * ones(1, 40); % learning rate 
NMS_Threshold = 0.4; % non maximum suppression
Lambda = [0.5 0.1 4];

%% Main
for i = 1:length(DatasetNameList)
    DatasetName = DatasetNameList{i};
    ClassNameList = dir(ImageDir{i});
    ClassNameList(1:2) = [];
    ClassNameList = {ClassNameList.name};
    TempFinalResultSaveDir = New_mkdir(FinalResultSaveDir{i});
    for j = 1:length(ClassNameList)
        ClassName = ClassNameList{j};
        FinalResultSaveName = [TempFinalResultSaveDir '/' ClassName '.mat'];
        if exist(FinalResultSaveName, 'file')
            continue
        end
        imdb = [];
         
        opts = DeepInstCoseg(imdb, 'DatasetDir', DatasetDir, 'DatasetName', DatasetName, ...
            'GPUID', GPUID, 'ClassName', ClassName, 'Lambda', Lambda, 'RootDir', RootDir, 'SalDir', SalDir{i}, ...
            'learningRate', learningRate);
        close all
        opts = RunPeakBackPrapagation(imdb, opts);
        opts.SelectionParams = SelectionParams;
        opts.NMS_Threshold = NMS_Threshold;
        [AP, SelectProposals, SelectScores, GTInstMasks, Images, ImageName] = RunInstMaskGen(imdb, opts);
        save([TempFinalResultSaveDir '/' ClassName '.mat'], ...
            'AP', 'SelectProposals', 'SelectScores', 'GTInstMasks', 'Images', 'ImageName');
        close all
    end
end
