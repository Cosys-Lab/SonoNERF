classdef FourierEmbeddingLayer < nnet.layer.Layer % ...

    properties
        numFourierComponents
    end

    properties (Learnable)
    end

    methods
        function obj = FourierEmbeddingLayer(numFourierComponents)
            obj.numFourierComponents = numFourierComponents;
        end


        function [Z ] = predict(obj,X)
            numInputs = size( X, 1 );

            a = 0 : (obj.numFourierComponents-1);
            b = repelem(a, numInputs)*pi;
            
            % Z = [ sin( repmat( X, obj.numFourierComponents, 1 ) .* b' ) ];
            Z = [ X ; sin( repmat( X, obj.numFourierComponents, 1 ) .* b' ) ; cos( repmat( X, obj.numFourierComponents, 1 ) .* b' ) ];
            
        end

    end
end