classdef Conv < dagnn.Filter
    properties
        size = [0 0 0 0]
        hasBias = true
        opts = {'cuDNN'}
        EnblePeakBack = false;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if ~obj.hasBias, params{2} = [] ; end
            outputs{1} = vl_nnconv(...
                inputs{1}, params{1}, params{2}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if ~obj.EnblePeakBack
                [derInputs, derParams] = Conv_backward(obj, inputs, params, derOutputs);
            else
                [derInputs, derParams] = PRM_backward(obj, inputs, params, derOutputs);
            end
        end
        
        function [derInputs, derParams] = Conv_backward(obj, inputs, params, derOutputs)
            if ~obj.hasBias, params{2} = [] ; end
            [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
                inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
        end
        
        function [derInputs, derParams] = PRM_backward(obj, inputs, params, derOutputs)
            offset = min(inputs{1}(:));
            Input_Offset = inputs{1} - offset;
            pos_weight = vl_nnrelu(params{1});
            norm_factor = vl_nnconv(...
                Input_Offset, pos_weight, [], ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
            eps = gpuArray(10^-10);
            zero_mask = norm_factor < eps;
            derOutputs{1} = derOutputs{1} ./ (abs(norm_factor) + eps);
            derOutputs{1}(zero_mask) = 0;
            derParams{1} = [];
            derParams{2} = [];
            
            [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
                Input_Offset, pos_weight, [], derOutputs{1}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
            derInputs{1} = Input_Offset .* derInputs{1};
        end
        
        function kernelSize = getKernelSize(obj)
            kernelSize = obj.size(1:2) ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
            outputSizes{1}(3) = obj.size(4) ;
        end
        
        function params = initParams(obj)
            % Xavier improved
            sc = sqrt(2 / prod(obj.size(1:3))) ;
            %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
            params{1} = randn(obj.size,'single') * sc ;
            if obj.hasBias
                params{2} = zeros(obj.size(4),1,'single') ;
            end
        end
        
        function set.size(obj, ksize)
            % make sure that ksize has 4 dimensions
            ksize = [ksize(:)' 1 1 1 1] ;
            obj.size = ksize(1:4) ;
        end
        
        function obj = Conv(varargin)
            obj.load(varargin) ;
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            obj.size = obj.size ;
            obj.stride = obj.stride ;
            obj.pad = obj.pad ;
        end
    end
end
