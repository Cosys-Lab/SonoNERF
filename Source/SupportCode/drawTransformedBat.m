function [hp] = drawTransformedBat(posBat, orientBat, batFaces, batVertices)
    
    batVerticesRot = ( rotz( orientBat(3) ) *roty( orientBat(2) ) * rotx( orientBat(1) ) * batVertices' )';
    batVerticesTrans = ( batVerticesRot' + posBat )';

    drawTriad( posBat, orientBat, 0.10 );
    hp = patch('faces', batFaces, 'vertices', batVerticesTrans, 'FaceColor', [ 0.65 0.13 0.13 ], 'EdgeAlpha', 0.1); 
        hp.FaceLighting = 'gouraud';
        hp.AmbientStrength = 1;
        hp.DiffuseStrength = 0.8;
        hp.SpecularStrength = 0.2;
        hp.SpecularExponent = 25;
        hp.BackFaceLighting = 'unlit';
end

