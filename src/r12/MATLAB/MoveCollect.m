function data=MoveCollect(varargin)
%MoveCollect(coordinates, mode, position, ser) moves the robot on a given path and collects data until the path is complete 
%Given coordinates as 'joint' or 'cart', moves to the position given
%Set mode to 'abs' or 'rel'
%
%For 'abs' setting:    data=MoveCollect('abs', mode, [WAIST SHOULDER ELBOW L-HAND WRIST], ser)
%
%For 'cart' setting:   data=MoveCollect('cart', mode, location, ser)
coord=varargin{1};
mode=varargin{2};             %Choose absolute or relative motion
ser=varargin{nargin};       %Serial port
set(ser, 'TimeOut', 0.1);    %Set timeout to 0.1 seconds (so sampling frequency of 10 Hz)

switch coord
    case 'joint'
        switch mode
            case 'rel'
                instruction = 'JMR';    %Joint Move Relative
            case 'abs'
                instruction = 'JMA';    %Joint Move Absolute
        end
        
        motion=flip([varargin{3} 0]);    %Define the motion we're going to do in form [YAW WRIST HAND ELBOW SHOULDER WAIST]
        motion=num2str(motion);
        command=['DECIMAL ' motion ' ' instruction];
        
        send=sprintf([command '\r']);   %Add return to command
        fprintf(ser,send);              %Write command to the serial port
        
    case 'cart'
        position=varargin{3};
        point=position.point;
        angle=position.angle;
        if mode=='rel'                      %More complicated to move relative
            position0=CartWhere(ser);       %Get current location
            point=position0.point+point;	%Relative point
            angle=position0.angle+angle; 	%relative angle
        end
        
        %Multiplying by 10 formats values properly for ROBOFORTH
        command=[num2str(angle(3)*10)  ' '  num2str(angle(2)*10)  ' '  num2str(angle(1)*10)  ' '  num2str(point(3)*10)  ' '  num2str(point(2)*10)  ' '  num2str(point(1)*10)  ' CM'];  %Amalgamate command string
        
        send=sprintf([command '\r']);   %Add return to command
        fprintf(ser,send);              %Write command to the serial port
        
    otherwise
        error("Coordinates must be either 'abs' or 'cart'")
        return
end

%Collect data until it stops changing, indicating we've stopped moving
data=[];            %Initialise data vector
threshold=7;
cont=1;
difference1=0;
difference2=0;
i=0;

disp('Starting data recording...')
while cont
    i=i+1;
    
    %Get the data and average the readings
    readings=GetDataDemo;
    averages=[mean(readings(:,1));
              mean(readings(:,2));
              mean(readings(:,3));
              mean(readings(:,4));
              mean(readings(:,5));
              mean(readings(:,6));];
    
    data=[data averages];            %Return a matrix of these averages
    
    %Calculate differences at current position (difference1) and previous position (difference2)
    if i ~=1
        difference2=difference1;
        difference1=mean(data(:,i)-data(:,i-1));
        
        disp(['Diff1 = ' num2str(difference1) '  Diff2 = ' num2str(difference2) '  abs(diff1-diff2) = ' num2str(abs(difference2-difference1))])
        %Break if consecutive differences are the same
        if abs(difference2-difference1) < threshold
            cont=0;
        end
    end
    pause(0.1);
end
disp('Data recording finished')

    
%Clear queue of serial commands from robot while reading data
messages=num2str(fscanf(ser));
while ~contains(messages, 'OK')
    messages=[messages num2str(fscanf(ser))];
end
