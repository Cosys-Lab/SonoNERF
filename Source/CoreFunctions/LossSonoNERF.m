classdef LossSonoNERF <  nnet.layer.RegressionLayer
    properties
        handleSonoNERF
        handleERTFReturn
        indexERTFSmoother
        ERTFSmoother
        smoothingKernel
    end
    
    methods
        function layer = LossSonoNERF( name )
            % Create an L1LossLayer object with the specified name
            layer.Name = name;
            layer.Description = 'Custom L1 Loss Layer';
        end
        
        function loss = forwardLoss(layer, Y, T)
            loss = sum(abs(Y(:) - T(:)));
           
        end
        
    end
end
