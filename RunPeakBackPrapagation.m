function opts = RunPeakBackPrapagation(imdb, opts)

opts.ResSaveName = strrep(opts.expDir, opts.DatasetName, [opts.DatasetName '_InstRes/']);
opts.ResSaveName = New_mkdir(strrep(opts.ResSaveName, [opts.ClassName '/'], ''));
opts.ResSaveName = [opts.ResSaveName opts.ClassName '.mat'];
opts.Vis = false;
opts.LayerRemoveName = {'dagnn.Loss', 'dagnn.LRN', 'dagnn.Correlate' ...
    'dagnn.CorrelateCross', 'dagnn.CorrelateSelf', ...
    'dagnn.SalDiffSelf', 'dagnn.SalDiffCross'};
if exist(opts.ResSaveName, 'file')
    return;
end

if opts.Vis
    opts.FigureSaveDir = New_mkdir(strrep(opts.expDir, opts.DatasetName, [opts.DatasetName '_InstVis/']));
end

ModelList = dir([opts.expDir '/net-epoch-*.mat']);
if isempty(ModelList)
    return;
end
if isempty(imdb)
    [Images, ImageName] = SetupTrainDataset(opts);
end
NumImages = size(Images, 4);
net = load([opts.expDir '/net-epoch-' num2str(length(ModelList)) '.mat'], 'net');
GPUInfo = gpuDevice;
if GPUInfo.Index ~= opts.GPUID
    gpuDevice(opts.GPUID)
end
AvgImage = imresize(single(net.net.meta.normalization.averageImage), [opts.Resolution opts.Resolution]);
AvgImage = gpuArray(AvgImage);
ImageList = bsxfun(@minus, single(gpuArray(Images)), AvgImage);
net = dagnn.DagNN.loadobj(net.net);
net.mode = 'test' ;
net.move('gpu')
net.conserveMemory = false;
net = RemoveNetLayers(net, opts.LayerRemoveName);
if isa(net.layers(end).block, 'dagnn.Sigmoid')
    net.removeLayer(net.layers(end).name);
end


Sel = cellfun(@(x) isa(x,'dagnn.Conv'), {net.layers.block});
for j = find(Sel)
    net.layers(j).block.EnblePeakBack = true;
end
Sel = cellfun(@(x) isa(x,'dagnn.PlanePeakGen'), {net.layers.block});
net.layers(Sel).block.EnblePeakBack = true;
PRM = cell(1, NumImages);
CoSalMap = cell(1, NumImages); 
PeakMask = cell(1, NumImages);
DerVarName = net.layers(Sel).outputs;
for j = 1:NumImages
    tic
    disp(['Extract PRM: Image ' num2str(j) '/' num2str(NumImages)])
    net.eval({'input', ImageList(:,:,:,j)});
    CoSalMap{j} = gather(net.vars(net.getVarIndex('SalMap')).value);
    PeakMasks = net.layers(Sel).block.PeakMasks;
    PeakMask{j} = gather(PeakMasks);
    [Row, Col] = find(PeakMasks);
    net.mode = 'normal' ;
    if isempty(PRM{j})
        PRM{j} = cell(1, length(Row));
    end
    for k = 1:length(Row)
        TempPeakMasks = gpuArray.zeros(size(PeakMasks), 'single');
        TempPeakMasks(Row(k), Col(k)) = 1;
        net.prm_peak_backprop({},{DerVarName{1}, TempPeakMasks});
        FinalMap = gather(sum(net.vars(1).der, 3));
        if opts.Vis
            NormalizeFinalMap = 1 - exp(-FinalMap/std(FinalMap(:)));
            NormalizeFinalMap(NormalizeFinalMap < 10^-2) = 0;
            imagesc(NormalizeFinalMap)
            FigSaveName = sprintf([opts.FigureSaveDir '/' ImageName{j} '_Inst_%02d.jpg'], k);
            export_fig(FigSaveName)
        end
        [net.vars.der] = deal([]);
        PRM{j}{k} = FinalMap;
    end
    toc
end
save(opts.ResSaveName, 'PRM', 'CoSalMap', 'ImageName', 'PeakMask', '-v7.3');
end


function net = RemoveNetLayers(net, LayerRemoveName)
for i = 1:length(LayerRemoveName)
    Sel = cellfun(@(x) isa(x,LayerRemoveName{i}), {net.layers.block});
    if any(Sel)
        net.removeLayer( {net.layers(Sel).name})
    end
end
end


function [images, ImageName] = SetupTrainDataset(opts)
ImageList = dir([opts.ImageDir '/*.jpg']);
NumImages = length(ImageList);
images = zeros([opts.Resolution opts.Resolution 3 NumImages], 'uint8');
ImageName = cell(1, NumImages);
for i = 1:NumImages
    Image = imread([opts.ImageDir '/' ImageList(i).name]);
    if size(Image, 3) == 1
        Image = repmat(Image, [1 1 3]);
    end
    images(:,:,:,i) = imresize(Image, [opts.Resolution opts.Resolution]);
    [~, ImageName{i},~] = fileparts(ImageList(i).name);
end
end
