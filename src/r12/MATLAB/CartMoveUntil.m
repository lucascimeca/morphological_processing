function end_position=CartMoveUntil(varargin)
%[point angle] = CartMoveUntil(movemode, endmode, parameters, position, ser, ser_stop)
%set "movemode" to 'rel' or 'abs'
%
%ser is the robot serial port, ser_stop is the Stop Arduino
%
%Moves to final point and / or angle depending on what's given until the
%end condition is met.
%Currently, endmode supports:
%   collision:  Moves until the sensor is pressed.
%               Set parameter to the sensing threshold
%   exit:       Moves until the sensor is no longer pressed
%               Set parameter to the sensing threshold
%   time:       Moves for the specified time
%               Set parameter to the time in seconds
%   manual:     Moves until return key is pressed

movemode=varargin{1};
endmode=varargin{2};
goal_position=varargin{nargin-2};
ser=varargin{nargin-1};
ser_stop=varargin{nargin};


if movemode=='rel'
    position0=CartWhere(ser);                                   %Get current location
	goal_position.point=position0.point+goal_position.point;	%Relative point
	goal_position.angle=position0.angle+goal_position.angle; 	%relative angle
elseif movemode=='abs'
    %Do nothing, this is also valid
else
    error('Not a valid movement mode')
end

%Multiplying by 10 formats values properly for ROBOFORTH
command=[num2str(goal_position.angle(3)*10)  ' '  num2str(goal_position.angle(2)*10)  ' '  num2str(goal_position.angle(1)*10)  ' '  num2str(goal_position.point(3)*10)  ' '  num2str(goal_position.point(2)*10)  ' '  num2str(goal_position.point(1)*10)  ' CM'];  %Amalgamate command string

send=sprintf([command '\r']);   %Add return to command
fprintf(ser,send);              %Write command to the serial port

switch endmode
    case 'collision'
        threshold=varargin{3};              %Input parameter is threshold
        average=0;
        while average < threshold
            readings=GetDataDemo;           %Read the sensor
            average=mean(mean(readings));   %Average these sensor readings
        end
    case 'exit'
        threshold=varargin{3};              %Input parameter is threshold
        average=0;
        while average > threshold
            readings=GetDataDemo;           %Read the sensor
            average=mean(mean(readings));   %Average these sensor readings
        end
    case 'time'
        time=varargin{3};    %Input parameter is the delay time
        pause(time);
    case 'manual'
        input('Press enter to stop robot')
end

%disp('Sending stop!')
%End conditions are met, so send the stop command
fprintf(ser_stop,'%s', 0)

%Clear queue of serial commands from robot
messages=num2str(fscanf(ser));
while ~contains(messages, 'OK') && ~contains(messages, 'ABORTED')
    messages=[messages num2str(fscanf(ser))];
end

%disp(messages);

%Update the positions
SendCommand('COMPUTE',ser);
end_position=CartWhere(ser);