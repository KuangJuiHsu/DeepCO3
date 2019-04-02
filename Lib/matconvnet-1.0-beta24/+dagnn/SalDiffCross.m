classdef SalDiffCross < dagnn.ElementWise
    properties
        BatchSize = [];
        ImageArea = [];
    end
    methods
        function outputs = forward(obj, inputs, params)
            obj.BatchSize = size(inputs{1});
            obj.ImageArea =  obj.BatchSize(1) * obj.BatchSize(2);
            InputsAll = gpuArray.zeros(obj.ImageArea * obj.BatchSize(end), 1, obj.BatchSize(3), 1, 'single');
            for i = 1:obj.BatchSize(end)
                Start = (i - 1) * obj.ImageArea + 1;
                End =  i * obj.ImageArea;
                InputsAll(Start:End,:,:) = reshape(inputs{1}(:,:,:,i), [obj.ImageArea  1 obj.BatchSize(3)]);
            end
            outputs = obj.SalDiffFoward({InputsAll});
        end
        
        function outputs = SalDiffFoward(obj, inputs)
            [h,w,~,batchsize]=size(inputs{1});
            outputs{1} = gpuArray.zeros([h,w h * w batchsize], 'single');
            for i = 1:batchsize
                Input = inputs{1}(:,:,:,i);
                Output = (Input(:) - Input(:)') .^ 2;
                outputs{1}(:,:,:,i) = reshape(Output, [h,w h*w]);
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derParams = {} ;
            InputsAll = gpuArray.zeros(obj.ImageArea * obj.BatchSize(end), 1, obj.BatchSize(3), 1, 'single');
            for i = 1:obj.BatchSize(end)
                Start = (i - 1) * obj.ImageArea + 1;
                End =  i * obj.ImageArea;
                InputsAll(Start:End,:,:) = reshape(inputs{1}(:,:,:,i), [obj.ImageArea  1 obj.BatchSize(3)]);
            end
            derInputsAll = obj.SalDiffBackward({InputsAll}, derOutputs);
            derInputs{1} = gpuArray.zeros(obj.BatchSize, 'single');
            for i = 1:obj.BatchSize(end)
                Start = (i - 1) * obj.ImageArea + 1;
                End =  i * obj.ImageArea;
                derInputs{1}(:,:,:,i) = reshape(derInputsAll(Start:End,:,:), obj.BatchSize(1:3));
            end
        end
        
        function derInputs = SalDiffBackward(obj, inputs, derOutputs)
            [h,w,~,batchsize]=size(inputs{1});
            derInputs = gpuArray.zeros([h,w 1 batchsize], 'single');
            for i = 1:batchsize
                Input = inputs{1}(:,:,:,i);
                Diff = reshape(derOutputs{1}(:,:,:,i), [h*w h*w]) .* (Input(:) - Input(:)');
                derInputs(:,:,:,i) = reshape(4 * sum(Diff, 2), [h,w]);
            end
        end
        
        function obj = SalDiffCross(varargin)
            obj.load(varargin) ;
        end
    end
end