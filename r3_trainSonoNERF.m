
%%
    clear
    close all
    close(findall(groot, "Type", "figure"));

    path(pathdef);
    addpath( genpath( 'Source' ) )    
    
    filenameData = 'DataCalculated/PreparedData/DataSonoNERF_Training_UALogo_test';

    load( filenameData)

    fnSave = [ 'DataCalculated/TrainedSonoNERFs/sonoNerfTrained - UALogo.mat' ];

    zeroFractionKeep = 0.2;
%% Training part

    commentString = "Here we train a SonoNERF based on spectrogram, a complex reflection model, and learned ERTFs. No ERTF normalization. Remote operation.";

    scalerOutput =  max( structSonoNERFData.outputs(:) );
    structSonoNERFData.outputsScaled = structSonoNERFData.outputs / scalerOutput;

    layersFullSonoNERF = [  featureInputLayer(7,"Name","featureinput")                
                SonoNerfLayer( 600, structSonoNERFData.frequencies, -10, scalerOutput  );
                regressionLayer
         ];

    checkpointDir = [ 'DataCalculated/Checkpoints/' datestr( now, "yyyymmdd_HHMMss") ];
    mkdir( checkpointDir )


    % Preload the ERTF with the real ERTF of the micronycteris
    load( 'Data\HRTF_Typical_Bat');
    freqVecERTFArray = array_struct.freq_vec;
    azVecArray = array_struct.az_grid_vec;
    elVecArray = array_struct.el_grid_vec;
    ertfLeftArray = array_struct.left_struct.hrtf_gridded_array;
    ertfRightArray = array_struct.right_struct.hrtf_gridded_array;
    [ azMeshArray, elMeshArray ] = meshgrid( azVecArray, elVecArray );
    freqVecSonoNERF = layersFullSonoNERF(2).frequencySamples(1:end/2);
    directionsSonoNERFCart = layersFullSonoNERF(2).directionsSampling;
    [azVecSonoNERFRad, elVecSonoNERFRad, ~] = cart2sph( directionsSonoNERFCart(1,:), directionsSonoNERFCart(2,:), directionsSonoNERFCart(3,:) );
    azVecSonoNERF = rad2deg( azVecSonoNERFRad );
    elVecSonoNERF = rad2deg( elVecSonoNERFRad );

    for cntFreq = 1 : length( freqVecSonoNERF )
        curFreqSonoNERF = freqVecSonoNERF( cntFreq );
        [ ~, idxFreqInArray ] = min( abs( freqVecERTFArray - curFreqSonoNERF ) );
        curERTFLeft = squeeze( ertfLeftArray( :, :, idxFreqInArray ) ); 
        curERTFRight = squeeze( ertfRightArray( :, :, idxFreqInArray ) ); 
        curInterpolatorLeft = scatteredInterpolant( azMeshArray(:), elMeshArray(:), curERTFLeft(:) );
        curInterpolatorRight = scatteredInterpolant( azMeshArray(:), elMeshArray(:), curERTFRight(:) );
        curERTFforSonoNERFLeft = curInterpolatorLeft( azVecSonoNERF, elVecSonoNERF );
        curERTFforSonoNERFRight = curInterpolatorRight( azVecSonoNERF, elVecSonoNERF );
        layersFullSonoNERF(2).ERTFReal( :, cntFreq ) = curERTFforSonoNERFLeft(:);
        layersFullSonoNERF(2).ERTFReal( :, cntFreq + length( freqVecSonoNERF ) ) = curERTFforSonoNERFRight(:);
        layersFullSonoNERF(2).ERTFImag( :, cntFreq ) = curERTFforSonoNERFLeft(:);
        layersFullSonoNERF(2).ERTFImag( :, cntFreq + length( freqVecSonoNERF ) ) = curERTFforSonoNERFRight(:);
    end



    trainingoptsFull = trainingOptions('adam', ...
        'MaxEpochs', 150, ...
        'MiniBatchSize', 512, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod', 10, ...
        'LearnRateDropFactor', 0.97, ...
        'Plots', 'training-progress', ...
        'CheckpointFrequency', 100, ...
        'CheckpointPath', checkpointDir, ...
        'OutputFcn',@(x)makeLogVertAx(x, [true true] ), ...
        'OutputNetwork', 'best-validation-loss', ...
        'Shuffle', 'every-epoch', ...
        'ValidationFrequency',100, ...
        'ValidationPatience', inf );



    % Here, we try to throw away a couple of zero outputs. Because they tend to be overly represented...
    idxRandomizer = randperm( length( structSonoNERFData.outputsScaled ) );
    dataInputRandomized = structSonoNERFData.inputs(idxRandomizer,:);
    dataOutoutRandomized = structSonoNERFData.outputsScaled(idxRandomizer,:);
    idxZeroOutputs = find( dataOutoutRandomized(:,1) == 0 );
    idxNonZeroOutputs = find( dataOutoutRandomized(:,1) > 0 );
    dataInputNonZero = dataInputRandomized( idxNonZeroOutputs, : );
    dataOutputNonZero = dataOutoutRandomized( idxNonZeroOutputs, : );
    idxZeroOutputsSelection = idxZeroOutputs( randi(length(idxZeroOutputs), [ round(zeroFractionKeep*length(idxZeroOutputs) ), 1 ] ) );

    dataInputZerosSelected = dataInputRandomized( idxZeroOutputsSelection, :);
    dataOutputsZerosSelected = dataOutoutRandomized( idxZeroOutputsSelection, :);
    
    % Here, we divide into train and validation data:
    dataInputClean = [ dataInputNonZero ; dataInputZerosSelected ];
    dataOutputClean = [ dataOutputNonZero ; dataOutputsZerosSelected ];

    percentageTrain = 0.9;

    [idxTrain, idxVal, idxTest ] = dividerand( size(dataInputClean,1), percentageTrain, 1-percentageTrain, 0);
    
    dataInputTrain = dataInputClean(idxTrain, :);
    dataOutputTrain = dataOutputClean(idxTrain, :);
    dataInputValid = dataInputClean(idxVal, :);
    dataOutputValid = dataOutputClean(idxVal, :);

    trainingoptsFull.ValidationData =  {dataInputValid, dataOutputValid};

    netTrained = trainNetwork(dataInputTrain, dataOutputTrain, layersFullSonoNERF, trainingoptsFull );


%%    
    structSonoNERF = struct();
    structSonoNERF.SonoNERF = netTrained;
    structSonoNERF.layersFullSonoNERF = layersFullSonoNERF;
    structSonoNERF.trainingoptsFull = trainingoptsFull;
    structSonoNERF.trainingoptsWarmup = [];
    structSonoNERF.filenameData = filenameData;
    structSonoNERF.structSonoNERFData = structSonoNERFData;
    structSonoNERF.idxRandomizer = idxRandomizer;
    structSonoNERF.scalerOutput = scalerOutput;
    structSonoNERF.commentString = commentString;
    structSonoNERF.zeroFractionKeep = zeroFractionKeep;

    save( fnSave, "structSonoNERF", "filenameData" );
    
    send_mail_message( 'jan.steckel', 'SonoNERF Trainng finished on own PC', 'PC Office', [])
