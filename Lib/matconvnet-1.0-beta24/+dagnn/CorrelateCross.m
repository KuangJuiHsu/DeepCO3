classdef CorrelateCross < dagnn.Filter
  properties
    opts = {'cuDNN'}
    ImageArea = [];
    BatchSize = [];
  end

  methods
    function outputs = forward(obj, inputs, params)
      % Assume the same size
      obj.BatchSize = size(inputs{1});
      obj.ImageArea =  obj.BatchSize(1) * obj.BatchSize(2);
      InputsAll = gpuArray.zeros(obj.ImageArea * obj.BatchSize(end), 1, obj.BatchSize(3), 1, 'single');
      for i = 1:obj.BatchSize(end)
          Start = (i - 1) * obj.ImageArea + 1;
          End =  i * obj.ImageArea;
          InputsAll(Start:End,:,:) = reshape(inputs{1}(:,:,:,i), [obj.ImageArea  1 obj.BatchSize(3)]);
      end
      outputs = obj.CorrelateFoward({InputsAll});
    end

    function outputs = CorrelateFoward(obj, inputs)
        [h,w,d,batchsize]= size(inputs{1});
        inputsfilter = reshape(permute(inputs{1},[3,1,2,4]),1,1,d,h*w,[]);
        outputs{1} = gpuArray.zeros(h,w,h*w,batchsize,'single');
        for i=1:batchsize
            outputs{1}(:,:,:,i) = vl_nnconv(inputs{1}(:,:,:,i), inputsfilter(:,:,:,:,i),[]) ;
        end
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      InputsAll = gpuArray.zeros(obj.ImageArea * obj.BatchSize(end), 1, obj.BatchSize(3), 1, 'single');
      for i = 1:obj.BatchSize(end)
          Start = (i - 1) * obj.ImageArea + 1;
          End =  i * obj.ImageArea;
          InputsAll(Start:End,:,:) = reshape(inputs{1}(:,:,:,i), [obj.ImageArea  1 obj.BatchSize(3)]);
      end
      derInputsAll = obj.CorrelateBackward({InputsAll}, derOutputs);
      derInputs{1} = gpuArray.zeros(obj.BatchSize, 'single');
      for i = 1:obj.BatchSize(end)
          Start = (i - 1) * obj.ImageArea + 1;
          End =  i * obj.ImageArea;
          derInputs{1}(:,:,:,i) = reshape(derInputsAll(Start:End,:,:), obj.BatchSize(1:3));
      end
      
      derParams = {} ;
    end

    
    function derInputs = CorrelateBackward(obj, inputs, derOutputs)
        [h,w,d,batchsize]= size(inputs{1});
        inputsfilter = reshape(permute(inputs{1},[3,1,2,4]),1,1,d,h*w,[]);
        derInputs1 = gpuArray.zeros(h,w,d,batchsize,'single');
        derInputs2 = gpuArray.zeros(h,w,d,batchsize,'single');
        
        for i=1:batchsize
            [dI1, dI2, ~] = vl_nnconv(...
                inputs{1}(:,:,:,i), inputsfilter(:,:,:,:,i), [], derOutputs{1}(:,:,:,i)) ;
            derInputs1(:,:,:,i) = dI1;
            derInputs2(:,:,:,i) = permute(reshape(dI2,d,h,w,[]),[2 3 1 4]);
        end
        derInputs = derInputs1 + derInputs2;
    end

    function obj = CorrelateCross(varargin)
      obj.load(varargin) ;
    end
  end
end