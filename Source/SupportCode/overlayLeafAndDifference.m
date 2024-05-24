% close all


[ surfaceFaces, surfaceVertices ] = stlreadNonCleaning("Data/Models/leafAndDragonfly.stl");
    surfaceVertices = surfaceVertices - mean(surfaceVertices);
    surfaceVertices = surfaceVertices * roty(0); 
    surfaceVertices = surfaceVertices/ 500;    

figure
        % set( gcf, 'position', [ -3494         214         942         973])
        set( gcf, 'position', [ 2906         144        1233        1052])
            hpGT = patch('faces', surfaceFaces, 'vertices', surfaceVertices, 'FaceColor', [0.2 0.3 0.1], 'EdgeAlpha', 0.3); 
            lightangle(-45,30)
            hpGT.FaceLighting = 'gouraud';
            hpGT.AmbientStrength = 1;
            hpGT.DiffuseStrength = 0.8;
            hpGT.SpecularStrength = 0.2;
            hpGT.SpecularExponent = 25;
            hpGT.BackFaceLighting = 'unlit';    
            light('Position', [-1 0 1], 'Style', 'local');
            light('Position', [1 0.5 -0.5], 'Style', 'local');            
            axis equal
            axis tight
            
            grid on
            xlim( limitsXComparison )
            ylim( limitsYComparison );
            zlim( limitsZComparison );
            % set( gca, 'view', cameraView )
            set( gca, 'linewidth', 1.5)
            set( gca, 'fontsize', 14)       
            xlabel( 'X-axis (m)' )
            ylabel( 'Y-axis (m)' )
            zlabel( 'Z-axis (m)' )    
            % camlight 
            camlight        
        
            view([90 0]);

            % differenceMIP = squeeze(max( volumePredictedEnergyWithoutDFNorm, [], 1 ));
            % differenceMIP = squeeze(max( volumePredictedEnergyWithDFNorm, [], 1 ));

            differenceMIP = squeeze(max( energyDifference, [], 1 ));
            % differenceMIP = 20*log10( differenceMIP  +0.0001 );
            % differenceMIP = differenceMIP - min(differenceMIP(:));
            % differenceMIP = differenceMIP / max(differenceMIP(:));
            % differenceMIP = 1 - differenceMIP;
            


            differenceMIPInterpol = imresize( differenceMIP, size(differenceMIP)*6, 'bicubic');
            differenceMIPInterpol = smooth2( differenceMIPInterpol );
            
            
            yVecInterp = linspace( yVec(1), yVec(end), size( differenceMIPInterpol, 1 ) ) + 0.007;
            zVecInterp = linspace( zVec(1), zVec(end), size( differenceMIPInterpol, 2 ) );

            differenceMIPTensor = permute( cat( 3, differenceMIPInterpol, differenceMIPInterpol ), [1 3 2]);
            % differenceMIPTensor( differenceMIPTensor < 0.01) = nan;
    
            xSliceLocation = 0.05;

            hold on;
            % figure
            hSlice = slice([xSliceLocation xSliceLocation+0.001]',yVecInterp(:),zVecInterp(:),differenceMIPTensor,xSliceLocation,[],[]);
            hSlice.FaceAlpha = 'flat' ;
            hSlice.EdgeAlpha = 0;
            colormap hot
            
            heatAlphaMap = ( (differenceMIPInterpol+0.01) ).^0.22;
            heatAlphaMap = heatAlphaMap - min(heatAlphaMap(:) );
            heatAlphaMap = heatAlphaMap / max( heatAlphaMap(:) );
            heatAlphaMap = heatAlphaMap * 60;
            heatAlphaMap( heatAlphaMap < 10 ) = 0;
            hSlice.AlphaData = heatAlphaMap;
            hSlice.AlphaDataMapping = 'direct';

