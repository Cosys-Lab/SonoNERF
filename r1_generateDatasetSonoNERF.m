
%% Init:

    clear
    close all
    clc

    path(pathdef);
    addpath( genpath( 'Source' ) )
    addpath( genpath( 'SonoTraceLab' ) )
    rng(42)
%%   Load the surface and prepare the surface normals
    
    nFreqsSim = 14;
    structMeshPreparation = struct();
    structMeshPreparation.orientation = [ 0 0 0];
    structMeshPreparation.position = [ 0 0 0];
    structMeshPreparation.vertexScaling = 1/500;
    structMeshPreparation.FLIPNORMALS = 1;
    structMeshPreparation.vecFreqSim = linspace( 20e3, 150e3, nFreqsSim ); 
    structMeshPreparation.fileNameMesh = "Data/Models/leafAndDragonfly.stl";
    % structMeshPreparation.fileNameMesh = "Data/Models/tripleBall.stl";
    % structMeshPreparation.fileNameMesh = "Data/Models/UALogo.stl";

    fnSave = [ 'DataCalculated/DataSonoNerf_IRs_leafAndDragonfly_test.mat'];

    structMeshPreparation.BRDFTransitionPosition = 2;
    structMeshPreparation.BRDFTransitionSlope = 1;
    structMeshPreparation.BRDFExponentSpecular = linspace( 12,4,nFreqsSim);
    structMeshPreparation.BRDFExponentDiffractive = linspace( 70,70,nFreqsSim);
    structMeshPreparation.materialStrengthSpecular = 10*linspace( 1,0.7,nFreqsSim);
    structMeshPreparation.materialStrengthDiffractive = 0.02*linspace( 0.8,1,nFreqsSim);
    structMeshPreparation.materialSTransitionPosition = 2;
    structMeshPreparation.materialSTransitionSlope = 1;
    structMeshPreparation.precomputeCurvature = 1;
    
   structSurface = prepareMeshSurface( structMeshPreparation,1 );
%% Setup the structs for processing

    arraySingleEar =  generateCircularArray( 0.0015, 0.005, 1);
    nMicsSingleEar = size( arraySingleEar, 1 );
    arrayFinal = [ zeros( nMicsSingleEar, 1 )  arraySingleEar ] + [ -0.010 0.01 0.005];
    coordsReceivers = arrayFinal;
    
    structSensor = struct();
    structSensor.position = [ 0.08 0 0];
    structSensor.orientation = [ 0 0 180]';
    structSensor.coordsEmitter = [ 0 0 -0.01];
    structSensor.coordsReceivers = coordsReceivers;
    structSensor.nMics = size( structSensor.coordsReceivers, 1 );
 
    % % Struct for the parameters of the simulation
    structSimulationParameters = struct();
    structSimulationParameters.doPlot = 0;
    structSimulationParameters.numSamplesImpresp = 7500;
    structSimulationParameters.sampleRateImpresp = 1e6;
    structSimulationParameters.limitsAzimuth = [-40 40];
    structSimulationParameters.limitsElevation = [-40 40];
    structSimulationParameters.numberOfDirections = 300000;
    structSimulationParameters.numberOfDirectionsPerCall = 150000;
    structSimulationParameters.vecFreqSim = structMeshPreparation.vecFreqSim;
    structSimulationParameters.numSamplesIRFilter = 256;
    structSimulationParameters.IRFilterGaussAlpha = 5;
    structSimulationParameters.numDiffractionPoints = 10000;
    structSimulationParameters.approximateImpulseResponseCutDB = -90;
    structSimulationParameters.approximateImpulseResponse = 0;
    structSimulationParameters.ditherRaytracing = 1;
    structSimulationParameters.speedOfSound = 343;
    %% Now calculate the whole setup:

    numSamplePoints = 400;
    radiusSphere = 0.4;
    centerSphere = [ 0 0 0];
    
    
    pointsSampleRaw = eq_point_set( 2, numSamplePoints );
    % pointsSampleRaw = pointsSampleRaw( :, pointsSampleRaw(1,:)>0);
    
    pointsSphere = ( radiusSphere + 0*rand(1,numSamplePoints)*radiusSphere) .* pointsSampleRaw + centerSphere(:);
    lookVectors = centerSphere(:) - pointsSphere/radiusSphere;
    nPosBat = size( pointsSphere, 2 );
    
    dataStorageMatrix = {};
    pointsBat = pointsSphere;
    PB = ProgressBar( nPosBat, 'Running Processing', 'cli');

    for cntPosBat = 1 : 1: nPosBat
        [ azLook, elLook, rLook ] = cart2sph( lookVectors(1, cntPosBat ), lookVectors(2, cntPosBat ), lookVectors(3, cntPosBat ) );
        
        dx =  lookVectors(1, cntPosBat );
        dy =  lookVectors(2, cntPosBat );
        dz =  lookVectors(3, cntPosBat );
        
        yaw = rad2deg( atan2( dy, dx+0.00001 ) ) + 15*randn();
        pitch = rad2deg( -atan2(dz, sqrt(dx^2+dy^2) ) ) + 15*randn();
     
        localStructSensor = structSensor;
        localStructSensor.position = [pointsBat(1,cntPosBat) pointsBat(2,cntPosBat) pointsBat(3,cntPosBat) ];
        localStructSensor.orientation = [ 0 pitch yaw]';
        
        % tic
        structSimulationResult = calculateImpulseResponseFast( localStructSensor, structSurface, structSimulationParameters );
        % toc

        structSimulatorOutput = struct();
        structSimulatorOutput.impulseResponse = structSimulationResult.impulseResponse;
        structSimulatorOutput.impulseResponseDiffraction = structSimulationResult.impulseResponseDiffraction;
        structSimulatorOutput.impulseResponseRaytracing = structSimulationResult.impulseResponseRaytracing;
        structSimulatorOutput.sensorInfo = localStructSensor;
        structSimulatorOutput.pointsBat = pointsBat;
        structSimulatorOutput.curPointBat = pointsBat( :, cntPosBat );
        structSimulatorOutput.lookVectors = lookVectors;

        dataStorageMatrix{ cntPosBat } = structSimulatorOutput;


        count( PB);
        % figure(1231)
        %     cla
        %         hp = patch('faces', structSurface.surfaceFaces, 'vertices', structSurface.surfaceVertices, 'FaceColor', [ 0.2 0.3 0.3 ], 'EdgeAlpha', 0.3); 
        %         % axis equal;
        %         xlabel( 'X-Axis' )
        %         ylabel( 'Y-Axis' )
        %         zlabel( 'Z-Axis' )
        %         grid on
        %         set( gca, 'view', [67.1313 7.3250] );
        %         axis equal
        % 
        %         hold on;
        %             drawTriad( localStructSensor.position(:), localStructSensor.orientation(:), 0.15)
        %         hold off;
        % xlim( [ -0.5 0.5])
        % ylim( [ -0.5 0.5])
        % zlim( [-0.5 0.5])
        % pause(0.1)

    end

    save( fnSave, 'dataStorageMatrix', '-v7.3' );
