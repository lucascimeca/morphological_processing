function [X, Y, readings]=XYMap(rows, columns, ser, ser_stop)
%[X, Y, readings]=XYMap(ser, ser_stop) guides the arm across the phantom
%Returns a map of the phantom, with the average sensor readings at each
%point and plots this as a 3D graph

initial.point=[0;300;-50];      %Point somewhere above the phantom, in the same x line as an inclusion
initial.angle=[90;-30;0];       %What angle do we want to stroke at? [90;-30;0] is vertical with flat edge leading

MoveToCart('set', initial, ser);

SetSpeed(1000,ser);

move_down.point=[0;0;-100]; move_down.angle=[0;0;0]; %Arbitrary point below the phantom which we aim to get to

%position=CartMoveUntil('rel', 'collision', 9000, move_down, ser, ser_stop); %Move down until the average sensor reading is 9000 so we've collided
position=CartMoveUntil('rel', 'manual', move_down, ser, ser_stop);  %For now, just do it manually


pause(1);


move_forward.point=[0;100;0]; move_forward.angle=[0;0;0];   %Arbitray point in front of the phantom which we aim to get to
%position=CartMoveUntil('rel', 'exit', 9000, move_forward, ser, ser_stop); %Move down until the average sensor reading is below 9000 so we've collided
position=CartMoveUntil('rel', 'manual', move_forward, ser, ser_stop);  %For now, just do it manually


pause(1);

%Move to position at the start of the probing
move_left=CartWhere(ser);
move_left.point=move_left.point+[-30;-20;0];
MoveToCart(move_left,ser);


%We want to probe 15cm lengths, across a 6cm width
%Do it in 12 strokes, with 5mm pitch, probing every 5mm
X=[];
Y=[];
readings=[];

line_start=CartWhere(ser);


for i=1:rows
    position=line_start;
    
    for j=1:columns
               
        disp(['Line ' num2str(i) '       Datapoint ' num2str(j)]);        
        %Update data variables
        X=[X position.point(1)];
        Y=[Y position.point(2)];
        
        %For now, just collect random data
        data_average=rand(1);
        
        %data=GetDataTriangle;
        %data_average=mean(mean(data));
        readings=[readings data_average];
        
        %Define the next point and go there, via an intermediate point above
        position.point=position.point+[0;-5;0];
        position.angle=position.angle;
        
        above.point=position.point+[0;0;10];
        above.angle=position.angle;
        
        SetSpeed(500,ser);
        MoveToCart(above,ser);
        SetSpeed(100,ser);
        MoveToCart(position,ser);
    end
    
    %Move up
    position.point=position.point+[0;0;50];
    SetSpeed(5000,ser);
    MoveToCart(position,ser);
    
    %Go back to point 5mm right of start of row
    line_start.point=line_start.point+[5;0;0];
    
    %Add another point above the start so we probe vertically
    line_start_above=line_start;
    line_start_above.point=line_start_above.point+[0;0;10];
    
    MoveToCart(line_start_above, ser);
    SetSpeed(100, ser);
    MoveToCart(line_start, ser);
end
