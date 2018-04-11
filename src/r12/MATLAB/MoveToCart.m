function messages=MoveToCart(varargin)
%MoveToCart(mode, position, ser)
%Move the head to the cartesian coordinates defined by position.point and angle defined by position.angle
%Point is a vector [x y z] in mm
%Angle is another vector [pitch roll yaw] in degrees
%Pass something to 'set' if you want to set to cartesian mode we want to set the robot to cartesian mode

ser=varargin{nargin};
position=varargin{nargin-1};

if nargin==3        %Change to cartesian mode if not already in it
    SendCommand('cartesian', ser);
end

point=position.point;
angle=position.angle;

%Multiplying by 10 formats values properly for ROBOFORTH
command=[num2str(angle(3)*10)  ' '  num2str(angle(2)*10)  ' '  num2str(angle(1)*10)  ' '  num2str(point(3)*10)  ' '  num2str(point(2)*10)  ' '  num2str(point(1)*10)  ' CM'];  %Amalgamate command string

messages=SendCommand(command,ser);