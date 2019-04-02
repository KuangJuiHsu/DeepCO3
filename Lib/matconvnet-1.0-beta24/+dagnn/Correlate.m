classdef Correlate < dagnn.Filter
  properties
    opts = {'cuDNN'}
    Numpair = []
  end

  methods
    function outputs = forward(obj, inputs, params)
      % Assume the same size
      [h,w,d,batchsize]= size(inputs{1});
      obj.Numpair = batchsize / 2;
      Input1 = inputs{1}(:,:,:,1:obj.Numpair);
      Input2 = inputs{1}(:,:,:,obj.Numpair+1:end);
      inputs2filter = reshape(permute(Input2,[3,1,2,4]),1,1,d,h*w,[]);
      outputs{1} = gpuArray.zeros(h,w,h*w,obj.Numpair,'single');
      for i=1:obj.Numpair
        outputs{1}(:,:,:,i) = vl_nnconv(Input1(:,:,:,i), inputs2filter(:,:,:,:,i),[]) ;
      end
      outputs{1} = reshape(outputs{1}, [h w h w obj.Numpair]);
      
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [h,w,d,~]= size(inputs{1});
      derOutputs{1} = reshape(derOutputs{1}, [h w h*w obj.Numpair]);
      Input1 = inputs{1}(:,:,:,1:obj.Numpair);
      Input2 = inputs{1}(:,:,:,obj.Numpair+1:end);
      inputs2filter = reshape(permute(Input2,[3,1,2,4]),1,1,d,h*w,[]);
      derInputs1 = gpuArray.zeros(h,w,d,obj.Numpair,'single');
      derInputs2 = gpuArray.zeros(h,w,d,obj.Numpair,'single');

      for i=1:obj.Numpair
      [dI1, dI2, ~] = vl_nnconv(...
        Input1(:,:,:,i), inputs2filter(:,:,:,:,i), [], derOutputs{1}(:,:,:,i)) ;
        derInputs1(:,:,:,i) = dI1;
        derInputs2(:,:,:,i) = permute(reshape(dI2,d,h,w,[]),[2 3 1 4]);
      end
      derInputs{1} = cat(4, derInputs1, derInputs2);
      derParams = {} ;
    end

    function obj = Correlate(varargin)
      obj.load(varargin) ;
    end
  end
end

