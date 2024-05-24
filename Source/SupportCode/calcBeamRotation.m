 

function rotatedVector= calcBeamRotation( beamDirection, inputVector )


    rotatedVector = rotz( beamDirection(3) ) *roty( beamDirection(2) ) * rotx( beamDirection(1) ) * inputVector; 


end
