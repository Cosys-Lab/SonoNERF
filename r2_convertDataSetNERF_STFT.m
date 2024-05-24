
%% Init
    clear
    path(pathdef);


    addpath( genpath( 'Source' ) )
    
    rng(42)
%% Load
    load( 'Data\HRTF_Typical_Bat');
    M = matfile('DataCalculated\Simulations/DataSonoNerf_IRs_UALogo_public.mat');
    fnSave = 'DataCalculated/PreparedData/DataSonoNERF_Training_UALogo_test.mat';    

%% Settings
    freqStart = 10e3;
    freqEnd = 100e3;
    rangeStart = 0.15;
    rangeEnd = 0.75;

%% Pre-extractions    
    dataStorageMatrix = {};


    stftLength = 64;
    stftOverlap = 60;
    stftNFFT = 512;
    sigEmit =[0 1 0 ];

    dbCut = -10;

    outputOfMEval = evalc('M');
    pattern = 'dataStorageMatrix:\s+\[\d+x(\d+)\s+cell\]';
    matches = regexp(outputOfMEval, pattern, 'tokens');
    if ~isempty(matches)
        numMeasurements = str2double(matches{1}{1});
    else
        numMeasurements = -1;
        error( 'File for the data is not found' )
    end

    PB = ProgressBar( numMeasurements, 'Calculating data', 'cli');
    parfor cntMeasurement = 1 : numMeasurements 
    
        currentData = M.dataStorageMatrix(1,cntMeasurement);
        currentData = currentData{1};
        irSimulated = currentData.impulseResponse;

        irFilteredLeft = zeros( size( irSimulated ) );
        irFilteredRight = zeros( size( irSimulated ) );
        
        for cntChannel = 1 : size( irSimulated, 2 )
            curFIRLeft = array_struct.left_struct.FIR_coeffs( cntChannel, : );
            curFIRRight = array_struct.right_struct.FIR_coeffs( cntChannel, : );
            irFilteredLeft( :, cntChannel ) = filter( curFIRLeft, 1, conv( irSimulated( :, cntChannel ), sigEmit, 'same' ) );

            irFilteredRight( :, cntChannel ) = filter( curFIRRight, 1,  conv( irSimulated( :, cntChannel), sigEmit, 'same' ) );
        end
        
        irOutputLeft = sum( irFilteredLeft, 2 );
        irOutputRight = sum( irFilteredRight, 2 );

        [leftSTFT, freqsSTFT, timeSTFT ] = spectrogram( irOutputLeft, stftLength,stftOverlap,stftNFFT,1e6, 'yaxis');
        [rightSTFT, ~, ~] = spectrogram( irOutputRight, stftLength,stftOverlap,stftNFFT,1e6, 'yaxis');

        rangeSTFT = timeSTFT * 343 / 2;

        idxFreqStart = get_vec_idx( freqStart, freqsSTFT );
        idxFreqEnd = get_vec_idx( freqEnd, freqsSTFT );

        idxRangeStart = get_vec_idx( rangeStart, rangeSTFT );
        idxRangeEnd = get_vec_idx( rangeEnd, rangeSTFT );

        sensorPosition = currentData.sensorInfo.position(:);
        sensorOrientation = currentData.sensorInfo.orientation(:);

        binauralSTFTSlice = abs( [ leftSTFT( idxFreqStart:idxFreqEnd, idxRangeStart:idxRangeEnd ) ; rightSTFT( idxFreqStart:idxFreqEnd, idxRangeStart:idxRangeEnd ) ] );
        binauralSTFTSlice = 20*log10(binauralSTFTSlice + 10^(dbCut / 20 ) ) - dbCut;
        binauralSTFTSlice( binauralSTFTSlice < 0 ) = 0;

        rangeSlice = rangeSTFT( idxRangeStart : idxRangeEnd );
        freqSlice = [ freqsSTFT(idxFreqStart:idxFreqEnd) ; freqsSTFT(idxFreqStart:idxFreqEnd) ];

        figure(123); imagesc( rangeSlice, freqSlice/1000, binauralSTFTSlice)
        colorbar

        curNumSamples = length( rangeSlice );
        sensorPositionRepped = repmat( sensorPosition, 1, curNumSamples );
        sensorOrientationRepped = repmat( sensorOrientation, 1, curNumSamples );
        
        storageStruct = struct();
        storageStruct.binauralSTFTSlice = binauralSTFTSlice;
        storageStruct.rangeSlice = rangeSlice;
        storageStruct.sensorPositionRepped = sensorPositionRepped;
        storageStruct.sensorOrientationRepped = sensorOrientationRepped;
        storageStruct.frequencies = [ freqsSTFT(idxFreqStart:idxFreqEnd) ; freqsSTFT(idxFreqStart:idxFreqEnd) ];

        dataStorageMatrix{ cntMeasurement } = storageStruct;

        count( PB );
    end

%% Combination storage:

    testStruct = dataStorageMatrix{1};
    numSamplesTotal = length( testStruct.sensorOrientationRepped ) * length( dataStorageMatrix );
    numFrequencies = size( testStruct.binauralSTFTSlice, 1 );
    numSamplesSingle = length( testStruct.sensorOrientationRepped ); 
    
    inputStacked = nan( numSamplesTotal, 7 );
    spectraStacked = nan( numSamplesTotal, numFrequencies );
    
    dataChunkMatrix = {};
    for cntMeasurement = 1 : numMeasurements
        curStruct = dataStorageMatrix{cntMeasurement};
        
        curInputs = [ curStruct.sensorPositionRepped ; curStruct.sensorOrientationRepped ; curStruct.rangeSlice ];
        curOutputs = curStruct.binauralSTFTSlice;
        
        curIdxStart = ( cntMeasurement - 1 ) * numSamplesSingle + 1;
        curIdxEnd = curIdxStart + numSamplesSingle - 1;
        inputStacked( curIdxStart : curIdxEnd, : ) = curInputs';
        spectraStacked( curIdxStart : curIdxEnd, : ) = curOutputs';

        curDataChunk = struct();
        curDataChunk.inputData = curInputs;
        curDataChunk.outputData = curOutputs;
        dataChunkMatrix{cntMeasurement} = curDataChunk;
    end


%% Now put it in variables:
    structSonoNERFData = struct();
    structSonoNERFData.inputs = inputStacked;
    structSonoNERFData.outputs = spectraStacked;
    structSonoNERFData.frequencies = curStruct.frequencies;
    structSonoNERFData.dataChunkMatrix = dataChunkMatrix;

    save( fnSave, 'structSonoNERFData' )


































    



 