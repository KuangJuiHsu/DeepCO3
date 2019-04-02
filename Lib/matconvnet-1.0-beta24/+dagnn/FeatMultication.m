classdef FeatMultication < dagnn.ElementWise
  properties
  end

  methods
    function outputs = forward(obj, inputs, params)
        outputs{1} = inputs{1} .* inputs{2};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      NumChannels = [size(inputs{1}, 3) size(inputs{2}, 3)];
      derInputs{1} = derOutputs{1} .* inputs{2};
      derInputs{2} = derOutputs{1} .* inputs{1};
      if NumChannels(1) == 1
          derInputs{1} = sum(derOutputs{1} .* inputs{2}, 3);
      elseif NumChannels(2) == 1
          derInputs{2} = sum(derOutputs{1} .* inputs{1}, 3);
      end
      derParams = {} ;
    end

    function obj = FeatMultication(varargin)
      obj.load(varargin) ;
    end
  end
end
