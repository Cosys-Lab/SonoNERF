
%% Init:
    clear
    close all
    clc
    path(pathdef);
    addpath( genpath( 'Source' ) )

%% Load a sononerf, a 3D model, etc

    load('DataCalculated/TrainedSonoNERFs/sonoNerfTrained - UALogo.mat')
    [ surfaceFaces, surfaceVertices ] = stlreadNonCleaning("Data/Models/UALogo.stl");
    surfaceVertices = surfaceVertices - mean(surfaceVertices);
    surfaceVertices = surfaceVertices* roty(0); 
    surfaceVertices = surfaceVertices/ 35;
    thresholdEstimator = 27000;
    limitsXComparison = [-0.1 0.1];
    limitsYComparison = [ -0.25 0.22 ];
    limitsZComparison = [ -0.15 0.18];
    chunckSelectVec = [ 230 280 120];
    limitsX = [-0.3 0.4];
    limitsY = [ -0.4 0.4 ];
    limitsZ = [ -0.2 0.2];

    % load('DataCalculated/TrainedSonoNERFs/sonoNerfTrained - leafAndDragonFly.mat') % This is leaf with DF
    % [ surfaceFaces, surfaceVertices ] = stlreadNonCleaning("Data/Models/leafAndDragonfly.stl");
    % surfaceVertices = surfaceVertices - mean(surfaceVertices);
    % surfaceVertices = surfaceVertices * roty(0); 
    % surfaceVertices = surfaceVertices/ 500;
    % thresholdEstimator = 1000;
    % limitsXComparison = [-0.1 0.1];
    % limitsYComparison = [ -0.25 0.22 ];
    % limitsZComparison = [ -0.21 0.18];
    % chunckSelectVec = [ 230 280 120];
    % limitsX = [-0.3 0.4];
    % limitsY = [ -0.4 0.4 ];
    % limitsZ = [ -0.2 0.2];
    
    % load('DataCalculated/TrainedSonoNERFs/sonoNerfTrained - tripleBall.mat')
    % [ surfaceFaces, surfaceVertices ] = stlreadNonCleaning("Data/Models/tripleBall.stl");
    % surfaceVertices = surfaceVertices - mean(surfaceVertices);
    % surfaceVertices = surfaceVertices* roty(0); 
    % surfaceVertices = surfaceVertices/ 35;
    % thresholdEstimator = 20000;
    % limitsXComparison = [-0.1 0.1];
    % limitsYComparison = [ -0.15 0.18 ];
    % limitsZComparison = [ -0.15 0.15];
    % chunckSelectVec = [ 230 280 120];
    % limitsX = [-0.3 0.4];
    % limitsY = [ -0.4 0.4 ];
    % limitsZ = [ -0.2 0.2];
    
    [ batFaces, batVertices ] = stlreadNonCleaning("Data/Models/singleBall.stl" );
    batVertices = batVertices - mean(batVertices);
    batVertices = batVertices* rotz(-90);
    batVertices = batVertices* roty(-20);
    batVertices = batVertices / 300;
    batVertices = batVertices + [ 0 0 0.01];

%% Inference: preparation    

    frequenciesERTF = structSonoNERF.SonoNERF.Layers(2).frequencySamples(1:end/2);
    numFrequencies = length( frequenciesERTF );
    directionsERTF = structSonoNERF.SonoNERF.Layers(2).directionsSampling;    

%% Chuck Selection   
    
    numChunkPlots = length( chunckSelectVec );

    posesBat = zeros( numChunkPlots, 6 );
    for cntChunk = 1 : numChunkPlots
        dataChunk = structSonoNERF.structSonoNERFData.dataChunkMatrix{chunckSelectVec(cntChunk)};
        poseIn = dataChunk.inputData(1:6,1);
        posesBat( cntChunk, : ) = poseIn(:);
    end
    posBatAllMajor = posesBat( :, 1:3)';
    orientBatAllMajor = posesBat( :, 4:6)';

%% Show the original model with the poses for the chunks
    
    cameraView = [57.5755   30.5700];
    figPos = [-3394         228        1291         888];


    figure; 
        set( gcf, 'position', figPos )
        hp = patch('faces', surfaceFaces, 'vertices', surfaceVertices, 'FaceColor', [0.2 0.3 0.1], 'EdgeAlpha', 0.1); 


        hold on;
            for posBatCnt = 1 : numChunkPlots
                hpBat = drawTransformedBat(posBatAllMajor(:, posBatCnt), orientBatAllMajor( :, posBatCnt ), batFaces, batVertices);
                hpBat.FaceLighting = 'gouraud';
                hpBat.AmbientStrength = 1;
                hpBat.DiffuseStrength = 0.8;
                hpBat.SpecularStrength = 0.2;
                hpBat.SpecularExponent = 25;
                hpBat.BackFaceLighting = 'unlit';
            end
        hold off

        axis equal
        axis tight

        grid on
        xlim( limitsX )
        ylim( limitsY );
        zlim( limitsZ );
        set( gca, 'view', cameraView )
    
        lightangle(-45,30)
        hp.FaceLighting = 'gouraud';
        hp.AmbientStrength = 1;
        hp.DiffuseStrength = 0.8;
        hp.SpecularStrength = 0.2;
        hp.SpecularExponent = 25;
        hp.BackFaceLighting = 'unlit';    
        light('Position', [-1 0 1], 'Style', 'local');
        light('Position', [1 0.5 -0.5], 'Style', 'local');

        xlabel( 'X-axis (m)' )
        ylabel( 'Y-axis (m)' )
        zlabel( 'Z-axis (m)' )
            set( gca, 'linewidth', 1.5)
            set( gca, 'fontsize', 14)        
%% Inference: Spectrogram
     figure; 
     set( gcf, 'position', [-3791         743        2118         446]);
    for cntChunk = 1 : numChunkPlots
        dataTest = structSonoNERF.structSonoNERFData.dataChunkMatrix{ chunckSelectVec( cntChunk )};
        specgramPred = structSonoNERF.SonoNERF.predict( dataTest.inputData' )';
        
       
            subplot(2,numChunkPlots,cntChunk )
                imagesc(specgramPred)
                title('SonoNERF Prediction' )
            subplot(2,numChunkPlots,cntChunk + numChunkPlots )
                imagesc(dataTest.outputData / structSonoNERF.scalerOutput)
                title('Ground Truth' )
    end



%% Inference: Extract IsoSurface
    
    xVec = -0.3 : 0.005: 0.3;
    yVec = -0.1 : 0.005 : 0.1;
    zVec = -0.2 : 0.005 : 0.2;
    [ xGrid, yGrid, zGrid ] = meshgrid( yVec, xVec, zVec );

    directionsInterrogation = eq_point_set( 2, 100 );
    [ azInterrogation, elInterrogation, rInterrogation ] = cart2sph( directionsInterrogation(1,:), directionsInterrogation(2,:), directionsInterrogation(3,:) );
   
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

%% Render the full thing in 3D!!


    
    volumePredictedVecEnergy = sum( dataStorageMatrix, 2 );
    

    volumePredictedEnergy = reshape( volumePredictedVecEnergy,  [ length(xVec) length(yVec) length(zVec)] );

    volumePredictedEnergy = smooth3( volumePredictedEnergy, 'box', 9 );

    
    [faces,verts] = isosurface(xGrid, yGrid, zGrid, volumePredictedEnergy, thresholdEstimator);

    surfaceMeshExtracted = surfaceMesh( verts, faces );
    surfaceMeshSmooth = smoothSurfaceMesh(surfaceMeshExtracted,1);
    % surfaceMeshShow(surfaceMeshSmooth,Title="Extracted SonoNERF Surface")

    figure; 
        % set( gcf, 'position', [ -3494         214         942         973])
            hpRec = patch('faces',surfaceMeshSmooth.Faces,'vertices', surfaceMeshSmooth.Vertices,  'FaceColor', [0.2 0.3 0.1], 'EdgeAlpha', 0.1);
            lightangle(-45,30)
            hpRec.FaceLighting = 'gouraud';
            hpRec.AmbientStrength = 1;
            hpRec.DiffuseStrength = 0.8;
            hpRec.SpecularStrength = 0.2;
            hpRec.SpecularExponent = 25;
            hpRec.BackFaceLighting = 'unlit';    
            light('Position', [-1 0 1], 'Style', 'local');
            light('Position', [1 0.5 -0.5], 'Style', 'local');            
            axis equal
            axis tight
            
            grid on
            xlim( limitsXComparison )
            ylim( limitsYComparison );
            zlim( limitsZComparison );
            set( gca, 'view', cameraView )
            xlabel( 'X-axis (m)' )
            ylabel( 'Y-axis (m)' )
            zlabel( 'Z-axis (m)' )
            set( gca, 'linewidth', 1.5)
            set( gca, 'fontsize', 14)
 