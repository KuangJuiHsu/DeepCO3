classdef LPNormalization < dagnn.ElementWise
  properties
    massp = [];
    mass = [];
    epsilon = 1e-2
    p = 2;
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.massp = sum(inputs{1}.^obj.p,3) + obj.epsilon ;
      obj.mass = obj.massp.^(1/obj.p) ;
      outputs{1} = bsxfun(@rdivide, inputs{1}, obj.mass) ;

    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      dzdy = bsxfun(@rdivide, derOutputs{1}, obj.mass) ;
      tmp = sum(dzdy .* inputs{1}, 3) ;
      derInputs{1} = dzdy - bsxfun(@times, tmp, bsxfun(@rdivide, inputs{1}.^(obj.p-1), obj.massp)) ;
      derParams = {} ;
    end

    function obj = LPNormalization(varargin)
      obj.load(varargin) ;
    end
  end
end
