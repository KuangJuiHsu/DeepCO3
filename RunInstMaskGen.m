function [APAll, SelectProposals, SelectScores, GTInstMasks, Images, ImageName] = RunInstMaskGen(imdb, opts)
APAll = [];
if ~exist(opts.ResSaveName, 'file')
    return
end
if isempty(imdb)
    [Images, ImageName] = SetupTrainDataset(opts);
end

% Parameters
Alpha = opts.SelectionParams(1);
Gamma = opts.SelectionParams(2);
NMS_Threshold = opts.NMS_Threshold;
proposal_size_limit = [0.00002, 0.85];
contour_width = 5;
AP_Threshold = [0.25 0.5 0.75];
GPUInfo = gpuDevice;
if GPUInfo.Index ~= opts.GPUID
    gpuDevice(opts.GPUID)
end
Kernal = gpuArray(ones(contour_width, contour_width));

% Paths
opts.SaveDir = opts.ResSaveName(1:end-4);
opts.SaveDir = New_mkdir([strrep(opts.SaveDir, [opts.DatasetName '_InstRes/'], [opts.DatasetName '_InstResult/']) '/']);
opts.GTDir = strrep(opts.ImageDir, 'Image', 'Mask');
Result = load(opts.ResSaveName);
PRM = Result.PRM;
CoSalMap = Result.CoSalMap;

NumImages = length(PRM);
SelectProposals = cell(1, NumImages);
SelectScores = cell(1, NumImages);
for i = 1:NumImages
    SelectProposalSaveName = [opts.SaveDir '/' ImageName{i} '.mat'];
    
    if ~exist(SelectProposalSaveName, 'file')
        ImgSize = size(Images{i});
        disp(['Proposal Selection: Image ' num2str(i) '/' num2str(NumImages)])
        tic
        TempCoSalMap = gpuArray(CoSalMap{i});
        TempCoSalMap = imresize(TempCoSalMap, ImgSize(1:2));
        
        ImageArea = prod(ImgSize(1:2));
        BGMap = 1-TempCoSalMap;

        NumPeaks = length(PRM{i});
        TempPRM = imresize(gpuArray(cell2mat(reshape(PRM{i}, [1 1 NumPeaks]))), ImgSize(1:2));
        TempPRM = TempPRM .* TempCoSalMap;
        
        try
            [ScoreList, ProposalList] = ProposalSelection(opts, ImageName{i}, ...
                Alpha, Gamma, proposal_size_limit, Kernal, ...
                ImageArea, BGMap, TempPRM, ImgSize, NumPeaks);
        catch % save gpu memory
            [ScoreList, ProposalList] = ProposalSelection_SaveMemory(opts, ImageName{i}, ...
                Alpha, Gamma, proposal_size_limit, Kernal, ...
                ImageArea, BGMap, TempPRM, ImgSize, NumPeaks);
        end
        
        
        [ScoreList, Index] = sort(ScoreList, 'descend');
        ProposalList = ProposalList(:,:,Index);
        
        [Proposals, Scores] = NMS(ProposalList, ScoreList, NMS_Threshold);
        Proposals = gather(Proposals);
        Scores = gather(Scores);
        save(SelectProposalSaveName, 'Proposals', 'Scores', '-v7.3');
        SelectProposals{i} = Proposals;
        SelectScores{i} = Scores;
        toc
    else
        Res = load(SelectProposalSaveName);
        SelectProposals{i} = Res.Proposals;
        SelectScores{i} = Res.Scores;
    end
end
GTInstMasks = cell(1, NumImages);
NumberInstances = cell(1, NumImages);
for i = 1:NumImages
    ImgSize = size(Images{i});
    InstMaksImgList = dir([opts.GTDir '/' ImageName{i} '_InstID*.png']);
    GTInstMasks{i} = false([ImgSize(1:2) length(InstMaksImgList)]);
    for j = 1 : length(InstMaksImgList)
        GTInstMasks{i}(:,:,j) = imread([opts.GTDir '/' InstMaksImgList(j).name]);
    end
    NumberInstances{i} = length(InstMaksImgList);
end
if ~isfield(opts, 'Threshold') || isempty(opts.Threshold)
    opts.Threshold = 20;
end
[SelectProposals, SelectScores] = NoiseFilter(SelectProposals, SelectScores, opts.Threshold);
APAll = EvalCoSegAP(SelectProposals, SelectScores, GTInstMasks, AP_Threshold);
end

function [ScoreList, ProposalList] = ProposalSelection(opts, ImageName, ...
                Alpha, Gamma, proposal_size_limit, Kernal, ...
                ImageArea, BGMap, TempPRM, ImgSize, NumPeaks)
Proposals = load([strrep(opts.ImageDir, 'Image', 'MCG_fast') '/' ImageName '.mat' ], 'masks', 'scores');
Proposals = gpuArray(imresize(Proposals.masks, ImgSize(1:2), 'nearest'));
ProposalArea = sum(sum(Proposals, 1),2);
Index = ProposalArea >= (ImageArea * proposal_size_limit(1)) & ProposalArea <= (ImageArea * proposal_size_limit(2));
Proposals = Proposals(:,:,Index);
NumProposals = size(Proposals, 3);
try
    MorphologyGrad = imdilate(Proposals, Kernal) ~= imerode(Proposals, Kernal);
catch
    MorphologyGrad = gpuArray.false(size(Proposals));
    for j = 1:NumProposals
        MorphologyGrad(:,:,j) = imdilate(Proposals(:,:,j), Kernal) ~= imerode(Proposals(:,:,j), Kernal);
    end
end
ScoreList = gpuArray.zeros(NumPeaks, 1, 'single');
ProposalList = gpuArray.false([ImgSize(1:2) NumPeaks]);
for j = 1:NumPeaks
    try
        Score = Alpha * sum(sum(TempPRM(:,:,j) .* Proposals, 1), 2) ...
            + sum(sum(TempPRM(:,:,j) .* MorphologyGrad, 1), 2) ...
            - Gamma * sum(sum(Proposals .* BGMap, 1), 2);
    catch
        Score = gpuArray.zeros(NumProposals, 1, 'single');
        TTempPRM = TempPRM(:,:,j);
        for k = 1:NumProposals
            Score(k) = Alpha * sum(TTempPRM(Proposals(:,:,k))) ...
                + sum(TTempPRM(MorphologyGrad(:,:,k))) ...
                - Gamma * sum(BGMap(Proposals(:,:,k)));
        end
        
    end
    [ScoreList(j), Index] = max(Score);
    ProposalList(:,:,j) = Proposals(:,:,Index);
end
end


function [ScoreList, ProposalList] = ProposalSelection_SaveMemory(opts, ImageName, ...
                Alpha, Gamma, proposal_size_limit, Kernal, ...
                ImageArea, BGMap, TempPRM, ImgSize, NumPeaks)
ProposalStep = 1500;
Proposals = load([strrep(opts.ImageDir, 'Image', 'MCG_fast') '/' ImageName '.mat' ], 'masks', 'scores');
NumAllProposals = size(Proposals.masks, 3);
ProposalMasks = Proposals.masks;
BatchIndex = 1:ProposalStep:NumAllProposals;
ScoreList = cell(1, length(NumPeaks));
ProposalList = cell(1, length(NumPeaks));

for ii = BatchIndex
    Start = ii;
    End = min(Start + ProposalStep - 1, NumAllProposals);
    Proposals = gpuArray(imresize(ProposalMasks(:,:,Start:End), ImgSize(1:2), 'nearest'));
    ProposalArea = sum(sum(Proposals, 1),2);
    Index = ProposalArea >= (ImageArea * proposal_size_limit(1)) & ProposalArea <= (ImageArea * proposal_size_limit(2));
    Proposals = Proposals(:,:,Index);
    NumProposals = size(Proposals, 3);
    try
        MorphologyGrad = imdilate(Proposals, Kernal) ~= imerode(Proposals, Kernal);
    catch
        MorphologyGrad = gpuArray.false(size(Proposals));
        for j = 1:NumProposals
            MorphologyGrad(:,:,j) = imdilate(Proposals(:,:,j), Kernal) ~= imerode(Proposals(:,:,j), Kernal);
        end
    end
    for j = 1:NumPeaks
        try
            Score = Alpha * sum(sum(TempPRM(:,:,j) .* Proposals, 1), 2) ...
                + sum(sum(TempPRM(:,:,j) .* MorphologyGrad, 1), 2) ...
                - Gamma * sum(sum(Proposals .* BGMap, 1), 2);
        catch
            Score = gpuArray.zeros(NumProposals, 1, 'single');
            TTempPRM = TempPRM(:,:,j);
            for k = 1:NumProposals
                Score(k) = Alpha * sum(TTempPRM(Proposals(:,:,k))) ...
                    + sum(TTempPRM(MorphologyGrad(:,:,k))) ...
                    - Gamma * sum(BGMap(Proposals(:,:,k)));
            end
            
        end
        if ii == 1
            ScoreList{j} = gpuArray([]);
            ProposalList{j} = gpuArray([]);
        end
        [TempScore, Index] = max(Score);
        ScoreList{j} = cat(1, ScoreList{j}, TempScore);
        ProposalList{j} = cat(3, ProposalList{j}, Proposals(:,:,Index));
    end
end
NewScoreList = gpuArray.zeros(NumPeaks, 1, 'single');
NewProposalList = gpuArray.false([ImgSize(1:2) NumPeaks]);
for j = 1:NumPeaks
    [NewScoreList(j), ID] = max(ScoreList{j});
    NewProposalList(:,:,j) = ProposalList{j}(:,:,ID);
end
ScoreList = NewScoreList;
ProposalList = NewProposalList;
end


function [SelectProposals, SelectScores] = NoiseFilter(SelectProposals, SelectScores, Threshold)
NewScores = cell2mat(SelectScores(:));
Threshold = prctile(NewScores, Threshold);
for i = 1:length(SelectScores)
    Index = SelectScores{i} >= Threshold;
    if all(Index == 0)
        [~, Index] = max(SelectScores{i});
    end
    SelectScores{i} = SelectScores{i}(Index);
    SelectProposals{i} = SelectProposals{i}(:,:,Index);
end
end

function [images, ImageName] = SetupTrainDataset(opts)
ImageList = dir([opts.ImageDir '/*.jpg']);
% ImageList = ImageList(1:5);
NumImages = length(ImageList);
images = cell(1, NumImages);
ImageName = cell(1, NumImages);
for i = 1:NumImages
    Image = imread([opts.ImageDir '/' ImageList(i).name]);
    if size(Image, 3) == 1
        Image = repmat(Image, [1 1 3]);
    end
    images{i} = Image;
    [~, ImageName{i},~] = fileparts(ImageList(i).name);
end
end
