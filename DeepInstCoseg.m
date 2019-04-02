function [opts,imdb] = DeepInstCoseg(imdb, varargin)
opts.GPUID = 1;
opts.DatasetDir = [];
opts.DatasetName = [];
opts.expDir = [];
opts.ClassName = [];
opts.Resolution = 448;
opts.Lambda = [1 1];
opts.learningRate = 0.000001 * ones(1, 20);
opts.InitModelPath = [];
opts.RootDir = [];
opts.SalDir = [];
opts.solver = @adam;
opts.InitNetPath = [];
opts.RemoveLayerIndex = 34;
[opts, varargin] = vl_argparse(opts, varargin) ;
assert(~isempty(opts.SalDir), 'Provide Sal Data!!!')
if isempty(opts.expDir)
    opts.expDir = [opts.RootDir opts.DatasetName '/' opts.ClassName '/'];
end

if isempty(opts.InitModelPath)
    opts.InitModelPath = [opts.RootDir 'models/'];
end
New_mkdir(opts.expDir);
New_mkdir(opts.InitModelPath);
opts.InitModelName = [opts.InitModelPath '/InitModel.mat'];
opts.ImageDir = [opts.DatasetDir '/' opts.DatasetName '/Image/' opts.ClassName];
opts.ClassifierPath = [opts.RootDir '/models/imagenet-vgg-verydeep-16.mat'];


% training options (SGD)
opts.train.numSubBatches = 1;
opts.train.batchSize = 6;
opts.train.continue = false ;
opts.train.gpus = opts.GPUID;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = opts.learningRate;
opts.train.SaveEachEpoch = false;
opts.train.numEpochs = numel(opts.train.learningRate);
opts.train.derOutputs = {'SalLoss', 1, 'CopeakLoss', opts.Lambda(1), 'AffinityLoss', opts.Lambda(2)};
% the balanced weights presented in Eq. (1)


opts.train.solver = opts.solver;
opts = vl_argparse(opts, varargin) ;

IsModelExist = exist([opts.expDir '/' sprintf('net-epoch-%d.mat', numel(opts.train.learningRate))], 'file');


if ~IsModelExist || nargout == 2
    if isempty(imdb)
        imdb = SetupTrainDataset(opts);
    end
    if length(imdb.images.set) == 1
        return
    end
    train = find(imdb.images.set == 1);
    val = train;
    
    if IsModelExist
        return;
    end
    if ~exist(opts.InitModelName, 'file')
        Net = ModifyNet(opts);
        Net = Net.saveobj();
        save(opts.InitModelName, '-struct', 'Net', '-v7.3');
    end
    Net = load(opts.InitModelName);
    Net = dagnn.DagNN.loadobj(Net);
    AllImageIndex = train;
    AvgImage = imresize(single(Net.meta.normalization.averageImage), [opts.Resolution opts.Resolution]);
    FunHand = @(imdb,batch)getBatch(imdb, batch, AvgImage, AllImageIndex);
    if length(opts.Lambda) == 3
        Sel = cellfun(@(x) isa(x,'dagnn.AffinityLoss'), {Net.layers.block});
        Net.layers(Sel).block.Alpha = opts.Lambda(end);
    end
    save([opts.expDir '/opts.mat'], 'opts');
    cnn_train_dag(Net, imdb, FunHand, ...
        opts.train, ....
        'train', train, ...
        'val', val, ...
        opts.train);
end
end


function net = ModifyNet(opts)
if isempty(opts.InitNetPath)
    [net, FeatOutput] = fcnInitializeSaliencyModel('sourceModelPath', opts.ClassifierPath, 'RemoveLayerIndex', opts.RemoveLayerIndex) ;
    Sel = cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block}) | cellfun(@(x) isa(x,'dagnn.DropOut'), {net.layers.block});
    net.removeLayer({net.layers(Sel).name})
    net.renameVar('prediction', 'SalScore');
    net.addLayer('SigmoidNormalization', dagnn.Sigmoid(), 'SalScore', 'SalMap');
    net.addLayer('SalLoss', dagnn.WeightEuclidLoss, {'SalMap', 'SalImg'}, 'SalLoss');
    net.addLayer('SigmoidNormalization_2', dagnn.Sigmoid(), 'SmallScoreMap', 'SmallSalMap');
    net.rebuild();
    InputStr = 'SmallScoreMap';
else
    net = load(opts.InitNetPath);
    net = dagnn.DagNN.loadobj(net.net);
end
Sel = cellfun(@(x) isa(x,'dagnn.PlanePeakGen'), {net.layers.block});
if all(~Sel)
    net.addLayer('PlanePeakMapsGen', ...
        dagnn.PlanePeakGen(), InputStr, {'PosPlainPeakValues'});
end
net.addLayer('FeatNorm', dagnn.LPNormalization(), FeatOutput, 'FeatNorm');

net.addLayer('SelfFeatCorrelation', dagnn.CorrelateCross, 'FeatNorm', 'PixelAffinity');
net.addLayer('SelfSalCorrelation', dagnn.CorrelateCross, 'SmallSalMap', 'SalAffinity');
net.addLayer('SelfSalDiff', dagnn.SalDiffCross(), 'SmallSalMap', 'SalDiff');
net.addLayer('WeightPixelAffinityLoss', dagnn.AffinityLoss, {'SalAffinity', 'PixelAffinity', 'SalDiff'}, 'AffinityLoss');

net.addLayer('WeightFeatNorm', dagnn.FeatMultication(), {'FeatNorm', 'SmallSalMap'}, 'WeightFeatNorm');
net.addLayer('FeatCorrelation', dagnn.Correlate, {'WeightFeatNorm'}, 'FourDTensor');
net.addLayer('CoPeakGen', dagnn.CoPeakGen, {'FourDTensor'}, 'CoPeakValues');
net.addLayer('CopeakLoss', dagnn.CopeakLoss, {'CoPeakValues'}, 'CopeakLoss');
net.rebuild();
end


function y = getBatch(imdb, images, AvgImage, AllImageIndex)
if mod(length(images), 2) == 1
    ImageIndex = find(~ismember(AllImageIndex, images));
    RandomIndex= randperm(length(AllImageIndex) - length(images));
    images = [images ImageIndex(RandomIndex(1))];
end
AvgImage = gpuArray(AvgImage);
ImageList = bsxfun(@minus, single(gpuArray(imdb.images.images(:,:,:,images))), AvgImage);
SalImgList = im2single(gpuArray(imdb.images.SalImg(:,:,:,images)));
y = {'input', ImageList, 'SalImg', SalImgList};
end


function imdb = SetupTrainDataset(opts)
ImageList = dir([opts.ImageDir '/*.jpg']);
SalDir = [opts.SalDir '/' opts.ClassName '/'];
NumImages = length(ImageList);
imdb.images.images = zeros([opts.Resolution opts.Resolution 3 NumImages], 'uint8');
imdb.images.SalImg = zeros([opts.Resolution opts.Resolution 1 NumImages], 'uint8');
for i = 1:NumImages
    disp(['Reading Image: ' num2str(i)])
    Image = imread([opts.ImageDir '/' ImageList(i).name]);
    if size(Image, 3) == 1
        Image = repmat(Image, [1 1 3]);
    end
    imdb.images.images(:,:,:,i) = imresize(Image, [opts.Resolution opts.Resolution]);
    [~,ImgName,~] = fileparts(ImageList(i).name);
    SaImg = load([SalDir ImgName '.mat']);
    imdb.images.SalImg(:,:,:,i) = imresize(SaImg.SalMap, [opts.Resolution opts.Resolution]);
    
end
imdb.images.set = true(1, NumImages);
imdb.images.name = {ImageList.name};
end


function [net, FeatOutput] = fcnInitializeSaliencyModel(varargin)
NumClasses = 1;
opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
opts.RemoveLayerIndex = [];
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.sourceModelPath,'file')
  fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
  mkdir(fileparts(opts.sourceModelPath)) ;
  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', opts.sourceModelPath) ;
end
net = vl_simplenn_tidy(load(opts.sourceModelPath)) ;

net.meta.cudnnOpts = {'cudnnworkspacelimit', 512 * 1024^3} ;


if ~isempty(opts.RemoveLayerIndex)
    net.layers = net.layers(1:opts.RemoveLayerIndex-1);
end

net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.layers(5).block.pad = [0 1 0 1] ;
net.layers(10).block.pad = [0 1 0 1] ;
net.layers(17).block.pad = [0 1 0 1] ;
net.layers(24).block.pad = [0 1 0 1] ;
if opts.RemoveLayerIndex > 31
    net.layers(31).block.pad = [0 1 0 1] ;
end
if opts.RemoveLayerIndex > 32
    net.layers(32).block.pad = [3 3 3 3] ;
end

% remove the last layers to reduce the model size
switch opts.RemoveLayerIndex
    case 32
        InputDim = 512;
        FeatOutput = 'x31';
    case 34
        InputDim = 4096;
        FeatOutput = 'x33';
end
for i = 1:numel(net.layers)
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

net.addLayer('Conv1X1', dagnn.Conv('size', [1 1 InputDim 1]), net.layers(end).outputs, 'SmallScoreMap', {'Conv1X1_f', 'Conv1X1_b'});
f = net.getParamIndex('Conv1X1_f') ;
net.params(f).value = zeros(1, 1, InputDim, 1, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1 ;
f = net.getParamIndex('Conv1X1_b') ;
net.params(f).value = zeros(1, 1, 1, 1, 'single');
net.params(f).learningRate = 2;
net.params(f).weightDecay = 1 ;


upsample = 32;
filters = single(bilinear_u(upsample * 2, NumClasses, NumClasses)) ;
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', upsample, ...
  'crop', [upsample/2 upsample/2 upsample/2 upsample/2], ...
  'numGroups', NumClasses, ...
  'hasBias', false, ...
    'opts', net.meta.cudnnOpts), ...
  'SmallScoreMap', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.addLayer('objective', ...
  dagnn.Loss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

net.addLayer('accuracy', ...
   dagnn.Loss(), ...
  {'prediction', 'label'}, 'accuracy') ;
end
