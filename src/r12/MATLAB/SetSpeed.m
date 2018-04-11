function messages=SetSpeed(varargin)
%SetSpeed(%speed, ser)
%Sets the motor speed to speed. If not passed to the function, return to
%the default value of 10000
%Max speed is 32,767

%Check if we have been passed a speed or not.
if nargin==1        %No speed passed, so set to default
    speed=10000;
    ser=varargin{1};
else
    speed=varargin{1};  %Speed passed, so set to that speed
    ser=varargin{2};
end

if speed>32767                  %Check we're not above the speed limit
    error('Speed is too high!')
end


command=[num2str(speed) ' SPEED !'];

messages=SendCommand(command, ser);