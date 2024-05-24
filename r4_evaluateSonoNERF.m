
%% Init:
    clear
    close all
    clc
    path(pathdef);
    addpath( genpath( 'Source' ) )

    load('DataCalculated/TrainedSonoNERFs/sonoNerfTrained - UALogo.mat')

%% Inference: preparation    

    frequenciesERTF = structSonoNERF.SonoNERF.Layers(2).frequencySamples(1:end/2);
    numFrequencies = length( frequenciesERTF );
    directionsERTF = structSonoNERF.SonoNERF.Layers(2).directionsSampling;    

%% Inference: Spectrogram
    
    dataTest = structSonoNERF.structSonoNERFData.dataChunkMatrix{104};
    specgramPred = structSonoNERF.SonoNERF.predict( dataTest.inputData' )';
    
    figure; 
        subplot(1,2,1 )
            imagesc(specgramPred)
            title('SonoNERF Prediction' )
        subplot(1,2,2 )
            imagesc(dataTest.outputData / structSonoNERF.scalerOutput)
            title('Ground Truth' )

%% Inference: ERTF Extraction

    [ azERTFSim, elERTFSim, ~ ] = cart2sph( directionsERTF(1,:), directionsERTF(2,:), directionsERTF(3,:) );
    azERTFSim = rad2deg( azERTFSim );
    elERTFSim = rad2deg( elERTFSim );
    
    [ azMesh, elMesh ] = meshgrid( [-90:1:90 ], [-90:1:90 ] );

    fullERTFSim = sqrt( structSonoNERF.SonoNERF.Layers(2).ERTFReal.^2 + structSonoNERF.SonoNERF.Layers(2).ERTFImag.^2 );

    storageERTF = zeros( [ size(azMesh) numFrequencies 2 ] );
    for cntFreq = 1 : numFrequencies    
        curSliceLeft = ( double( gather( fullERTFSim(:, cntFreq ) ) ) );
        curSliceRight = ( double( gather( fullERTFSim(:, cntFreq + numFrequencies ) ) ) );
        
        if( isa( curSliceLeft, 'dlarray' ) )
            curSliceLeft = extractdata( curSliceLeft );
            curSliceRight = extractdata( curSliceRight );
        end

        interpolatorERTFLeft = scatteredInterpolant( azERTFSim(:), elERTFSim(:), curSliceLeft(:) );
        interpolatorERTFRight = scatteredInterpolant( azERTFSim(:), elERTFSim(:), curSliceRight(:) );

        storageERTF( :,:,cntFreq,1) = interpolatorERTFLeft( azMesh, elMesh );
        storageERTF( :,:,cntFreq,2) = interpolatorERTFRight( azMesh, elMesh );
    end

    
    figure; 
        idxFreqDisp = round( linspace( 1, 47, 4 ) );
        for cntFreqDisp = 1 : 4
            plotIdx = (cntFreqDisp-1)*2 + 1;
            subplot(4,2,plotIdx)
                imagesc( squeeze( storageERTF(:,:,idxFreqDisp(cntFreqDisp),1)))
            subplot(4,2,plotIdx+1)
                imagesc( squeeze( storageERTF(:,:,idxFreqDisp(cntFreqDisp),2)))      
        end


%% Inference: Extract IsoSurface
    
    xVec = -0.3 : 0.005: 0.3;
    yVec = -0.1 : 0.005 : 0.1;
    zVec = -0.2 : 0.005 : 0.2;
    [ xGrid, yGrid, zGrid ] = meshgrid( yVec, xVec, zVec );

   
    % directionsInterrogation = structSonoNERF.SonoNERF.Layers(2).directionsSampling( :, 1 : 10 : end );

    directionsInterrogation = eq_point_set( 2, 100 );
    [ azInterrogation, elInterrogation, rInterrogation ] = cart2sph( directionsInterrogation(1,:), directionsInterrogation(2,:), directionsInterrogation(3,:) );

    % azInterrogationVec = deg2rad( [ 0 -20 20 ] );
    % elInterrogationVec = deg2rad( [ 0 -45 45 ] );
    % 
    % [ azInterrogation, elInterrogation ] = meshgrid( azInterrogationVec, elInterrogationVec );
    % azInterrogation = azInterrogation(:);
    % elInterrogation = elInterrogation(:);

   
    idxFreqCombiner = [ 1 : 10 ; 11 : 20 ; 21 : 30 ; 31 : 40 ];

     dataStorageMatrix = zeros( length(xGrid(:)), length( azInterrogation ) );

    for cntDirection = 1 : length( azInterrogation )
        cntDirection
        curAz = azInterrogation( cntDirection );
        curEl = elInterrogation( cntDirection );
        dataIn = gpuArray( [ xGrid(:) yGrid(:) zGrid(:) ones( size(zGrid(:) ) )*curEl ones( size(zGrid(:) ) )*curAz ] );
        volumePredictedVecRaw = gather( structSonoNERF.SonoNERF.Layers(2).modelSonoNerf.predict(dataIn) );
        % volumePredictedVecRaw = gather( net.Layers(2).modelSonoNerf.predict(dataIn) );
        volumePredictedVecComplex = volumePredictedVecRaw(:,1:end/2) + 1i*volumePredictedVecRaw(:,end/2+1:end);
        % volumePredictedVecEnergy = sqrt( sum( abs(volumePredictedVecComplex), 2 ) );

        % volumePredictedVecEnergy = sum( abs(volumePredictedVecComplex(:, idxFreqCombiner(4,:)) ), 2 );
        volumePredictedVecEnergy = sum( abs(volumePredictedVecComplex(:, :) ), 2 );
        dataStorageMatrix( :, cntDirection ) = volumePredictedVecEnergy;
    end

%%
    % volumePredictedVecEnergy = sqrt( sum( ( dataStorageMatrix.^2  ), 2 ) );

    % volumePredictedVecEnergy = prod( dataStorageMatrix / max(dataStorageMatrix(:)), 2 );
    % volumePredictedVecEnergy = max( dataStorageMatrix, [], 2 );
    volumePredictedVecEnergy = sum( dataStorageMatrix, 2 );
    

    volumePredictedEnergy = reshape( volumePredictedVecEnergy,  [ length(xVec) length(yVec) length(zVec)] );
    
    volumePredictedEnergy = smooth3( volumePredictedEnergy, 'box', 9 );
    volshow( volumePredictedEnergy ) 
    
    % 
    % thresholdEstimator = mode( xGrid(:) ) *300;
    % % thresholdEstimator = 15
    % [faces,verts] = isosurface(xGrid, yGrid, zGrid, volumePredictedEnergy, thresholdEstimator);
    % 
    % surfaceMeshExtracted = surfaceMesh( verts, faces );
    % surfaceMeshSmooth = smoothSurfaceMesh(surfaceMeshExtracted,5);
    % surfaceMeshShow(surfaceMeshSmooth,Title="Extracted SonoNERF Surface")
    % 
    % % figure; 
    %     p = patch('faces',surfaceMeshSmooth.Faces,'vertices', surfaceMeshSmooth.Vertices,  'FaceColor', [ 0.2 0.3 0.3 ], 'EdgeAlpha', 0.3);
    %     axis equal 
    %     grid on
    %     camlight
    %     xlabel( 'X-axis (m)' )
    %     ylabel( 'Y-axis (m)' )
    %     zlabel( 'Z-axis (m)' )

    