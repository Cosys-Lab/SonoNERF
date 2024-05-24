
classdef SonoNerfLayer < nnet.layer.Layer % ...
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        layersSonoNerf
        directionsSampling
        frequencySamples
        sampleRateAcoustic
        numDirections
        dbCutOutput
        scalerOutput
        ERTFReal
        ERTFImag    
    end

    properties (Learnable)
        modelSonoNerf
        overallGainFactor
        callSpectrum
    end

    methods
        function obj = SonoNerfLayer( numSamplePoints, frequencyPoints, dbCutOutput, scalerOutput )
            lgraph = layerGraph();
            tempLayers = featureInputLayer(5,"Name","featureinput");
            lgraph = addLayers(lgraph,tempLayers);
            
            tempLayers = FourierEmbeddingLayerExponentialSqrt(30);
            tempLayers.Name = "FourierEmbedding1";
            lgraph = addLayers(lgraph,tempLayers);
            
            tempLayers = [
                batchNormalizationLayer("Name","batchnorm_1")
                fullyConnectedLayer(256,"Name","fc")
                leakyReluLayer(0.01,"Name","relu")
                fullyConnectedLayer(256,"Name","fc_1")
                leakyReluLayer(0.01,"Name","relu_1")
                fullyConnectedLayer(256,"Name","fc_2")
                leakyReluLayer(0.01,"Name","relu_2")
                fullyConnectedLayer(256,"Name","fc_2_1")
                leakyReluLayer(0.01,"Name","relu_2_1")
                batchNormalizationLayer("Name","batchnorm_4_1")];
            lgraph = addLayers(lgraph,tempLayers);
            
            tempLayers = [
                depthConcatenationLayer(2,"Name","depthcat")
                fullyConnectedLayer(256,"Name","fc_2_2")
                leakyReluLayer(0.01,"Name","relu_2_2")
                fullyConnectedLayer(256,"Name","fc_2_3")
                leakyReluLayer(0.01,"Name","relu_2_4")
                batchNormalizationLayer("Name","batchnorm_4_3")];
            lgraph = addLayers(lgraph,tempLayers);
            
            tempLayers = FourierEmbeddingLayerExponential(20);
            tempLayers.Name = "FourierEmbedding2";
            lgraph = addLayers(lgraph,tempLayers);
            
            tempLayers = [
                depthConcatenationLayer(2,"Name","depthcat_1")
                fullyConnectedLayer(256,"Name","fc_2_4")
                leakyReluLayer(0.01,"Name","relu_2_3")
                fullyConnectedLayer(128,"Name","fc_2_5")
                leakyReluLayer(0.01,"Name","relu_2_5")
                fullyConnectedLayer( length(frequencyPoints)/2 * 2 ,"Name","fc_3") ]; % This is the layer that outputs. As frequency contains both left and right, it needs to be divided by 2, and then *2 for complex numbers.

            lgraph = addLayers(lgraph,tempLayers);
            lgraph = connectLayers(lgraph,"featureinput","FourierEmbedding1");
            lgraph = connectLayers(lgraph,"featureinput","FourierEmbedding2");
            lgraph = connectLayers(lgraph,"FourierEmbedding1","batchnorm_1");
            lgraph = connectLayers(lgraph,"FourierEmbedding1","depthcat/in1");
            lgraph = connectLayers(lgraph,"batchnorm_4_1","depthcat/in2");
            lgraph = connectLayers(lgraph,"FourierEmbedding2","depthcat_1/in2");
            lgraph = connectLayers(lgraph,"batchnorm_4_3","depthcat_1/in1");

            obj.layersSonoNerf = lgraph.Layers;
            obj.modelSonoNerf = dlnetwork( lgraph );

            pointsSampleRaw = eq_point_set( 2, 2*numSamplePoints );
            obj.directionsSampling = pointsSampleRaw( :, pointsSampleRaw(1,:) > 0 );
            
            obj.frequencySamples = frequencyPoints;
            obj.sampleRateAcoustic = 1e6;
            obj.ERTFReal = dlarray( ones( length( obj.directionsSampling ), length( obj.frequencySamples ) ) );
            obj.ERTFImag = dlarray( ones( length( obj.directionsSampling ), length( obj.frequencySamples ) ) );
            obj.Name = "SonoNERF";
            obj.numDirections = length( obj.directionsSampling ); 
            obj.dbCutOutput = dbCutOutput;
            obj.scalerOutput = scalerOutput;

            obj.overallGainFactor = dlarray( 1 );

            obj.callSpectrum = ones( length( obj.frequencySamples )/2, 1 );

        end

        % function obj = initialize(layer,layout)
        % 
        % end
        
        % function [ ERTFReal, ERTFImag ] = returnCurrentERTF( obj )
        %     ERTFReal = obj.ERTFReal;
        %     ERTFImag = obj.ERTFImag;
        % 
        % end
        % 
        function [ objRet ] = returnSonoNERFObject( obj )
            objRet = obj;

        end

        function [currentSonoNERFSpectrum] = predict( obj, nerfInput )
                       
            batchSize = size( nerfInput, 2 );

            sensorPosition = nerfInput( 1:3, : );
            sensorOrientation = deg2rad( nerfInput( 4:6, : ) );
            currentRangeSlice = nerfInput( 7, : );
            emptyZeros  = dlarray( zeros( 1, batchSize ) );
            emptyOnes = dlarray( ones( 1, batchSize ) );
            
            rotZmat = permute( cat(3, [cos(sensorOrientation(3, :)); -sin(sensorOrientation(3, :)); emptyZeros], [sin(sensorOrientation(3, :)); cos(sensorOrientation(3, :)); emptyZeros], [emptyZeros; emptyZeros; emptyOnes]), [3 1 2]);
            rotYmat = permute( cat(3, [cos(sensorOrientation(2, :)); emptyZeros; sin(sensorOrientation(2, :))], [emptyZeros; emptyOnes; emptyZeros], [-sin(sensorOrientation(2, :)); emptyZeros; cos(sensorOrientation(2, :))]), [3 1 2]);
            rotXmat = permute( cat(3, [emptyOnes; emptyZeros; emptyZeros], [emptyZeros; cos(sensorOrientation(1, :)); -sin(sensorOrientation(1, :))], [emptyZeros; sin(sensorOrientation(1, :)); cos(sensorOrientation(1, :))]), [3 1 2]);

            fullRot = pagemtimes( pagemtimes( rotZmat, rotYmat ), rotXmat ); 
            rotatedDirections = pagemtimes( fullRot, obj.directionsSampling );
            rotTimesRange = pagemtimes( rotatedDirections, reshape(currentRangeSlice, [1, 1, numel(currentRangeSlice ) ] ) );
            
            pointsSampling = permute( repmat( sensorPosition, 1, 1, obj.numDirections ), [1 3 2] ) + rotTimesRange;
 
            orientationSensorPermed = permute( repmat( sensorOrientation(2:3,:), 1, 1, obj.numDirections ), [1 3 2] );
            
            inputsSonoNerf = real( cat( 1, pointsSampling, orientationSensorPermed ) );
    
            
            inputsSonoNerf = reshape( inputsSonoNerf, [ size( inputsSonoNerf, 1 ) size( inputsSonoNerf, 2 ) * size( inputsSonoNerf, 3 ) ] );
            
            
            % Perform the NERF prediciton, scale it with the gain, make it complex, split it to binaural and fit it in the matrix for
            % subsequent processing.
            currentNerfPrediction = real( obj.modelSonoNerf.predict( inputsSonoNerf' ) );
            currentNerfPrediction = abs(obj.overallGainFactor) * currentNerfPrediction;  % Here we use abs() of the gain. It could also be 
            currentNerfPredictionComplex =  currentNerfPrediction(:, 1:end/2) + 1i*currentNerfPrediction(:,end/2+1:end);
            currentNerfPredictionBinaural = [ currentNerfPredictionComplex currentNerfPredictionComplex ]; %Nerf output is the same for left and right ear
            currentNerfPredictionT = ( permute( reshape( currentNerfPredictionBinaural,  [obj.numDirections batchSize length( obj.frequencySamples ) ] ), [ 1 3 2 ] ) );
          
            % Here we assume left and right distance to be equal, ie no ITD for now... Should update this
            curDeltaTime = real( 2*currentRangeSlice / 343 );
            currentRangeFunction = dlarray( exp( 2*pi*1i*obj.frequencySamples*curDeltaTime ) );
            currentRangeFunctionT = zeros( 1, size( currentRangeFunction, 1 ), batchSize );
            currentRangeFunctionT( 1, :, : ) = currentRangeFunction;
            currentRangeFunctionT = repmat( currentRangeFunctionT, [ obj.numDirections 1 1 ] );

            % Prepare the outerProduct of Nerf and Range
            outerProduct = ( currentNerfPredictionT .* currentRangeFunctionT );
            
            % Broken down complex multiplication of NERF/Range/ERTF
            a = real( outerProduct );
            b = real( imag( outerProduct ) );
            c = repmat(real( obj.ERTFReal), [1, 1, batchSize]);

            d =  repmat(real( obj.ERTFImag), [1, 1, batchSize]);
            e = (a .* c -b .* d);
            f = (b.*c + a.*d );
            eSummed = sum( e, 1 );
            fSummed = sum( f, 1 );
            
            % Transform the output to a logarithmic output, between 0 and maxDb, but then normalized to 1 
            currentSonoNERFSpectrum = abs( squeeze( sqrt( power( eSummed, 2) + power( fSummed, 2 ) ) ) ) / length(obj.directionsSampling);
        
            currentSonoNERFSpectrum = currentSonoNERFSpectrum ./ [ obj.callSpectrum ; obj.callSpectrum ];

            currentSonoNERFSpectrum = 20*log(currentSonoNERFSpectrum + 10^(obj.dbCutOutput / 20 ) )/log(10) - obj.dbCutOutput;
            currentSonoNERFSpectrum = currentSonoNERFSpectrum / obj.scalerOutput;

            % If batchsize is 1, do some transposes
            if( batchSize == 1 )
                currentSonoNERFSpectrum = currentSonoNERFSpectrum.';
            end
            
        end


    end
end
