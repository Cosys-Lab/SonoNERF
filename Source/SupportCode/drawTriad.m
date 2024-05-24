function drawTriad( vecPos, vecOrient, lengthTriad )

    triadRot = calcBeamRotation( vecOrient, eye(3) ) * lengthTriad;

    
    plot3( vecPos(1), vecPos(2), vecPos(3), '.' );
    

    p1 = [ vecPos ];
    
    p2 = [ vecPos + triadRot(:,1) ];
    plot3( [ p1(1) p2(1) ],[ p1(2) p2(2) ], [ p1(3) p2(3) ], 'r', 'linewidth', 5 ) 

    p2 = [ vecPos + triadRot(:,2) ];
    plot3( [ p1(1) p2(1) ],[ p1(2) p2(2) ], [ p1(3) p2(3) ], 'g', 'linewidth', 5 ) 

    p2 = [ vecPos + triadRot(:,3) ];
    plot3( [ p1(1) p2(1) ],[ p1(2) p2(2) ], [ p1(3) p2(3) ], 'b', 'linewidth', 5 ) 


    % p2 = [ vecPos + triadRot(1,:)' ];
    % plot3( [ p1(1) p2(1) ],[ p1(2) p2(2) ], [ p1(3) p2(3) ], 'r', 'linewidth', 5 ) 
    % 
    % p2 = [ vecPos + triadRot(2,:)' ];
    % plot3( [ p1(1) p2(1) ],[ p1(2) p2(2) ], [ p1(3) p2(3) ], 'g', 'linewidth', 5 ) 
    % 
    % p2 = [ vecPos + triadRot(3,:)' ];
    % plot3( [ p1(1) p2(1) ],[ p1(2) p2(2) ], [ p1(3) p2(3) ], 'b', 'linewidth', 5 ) 

end

