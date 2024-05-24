classdef SonoNERFfLossLayer < nnet.layer.RegressionLayer
    % This class implements the L1 loss layer for use in neural networks.

    methods
        function layer = SonoNERFfLossLayer(name)
            % Set layer name
            layer.Name = name;

            % Set layer description
            layer.Description = "Mean absolute error";
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Forward pass

            % Calculate L1 loss
            loss = ( sum( abs(Y(:)-T(:) ) ) + 4*sum( ( Y(:)-T(:) ).^2  ) ) / length( T(:) ) * 512;
        end
        % 
        % function dLdY = backwardLoss(layer, Y, T)
        %     % Backward pass
        % 
        %     % Compute gradients
        %     dLdY = sign(Y - T) / numel(Y);
        % end
    end
end
