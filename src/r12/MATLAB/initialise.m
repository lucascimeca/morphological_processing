function initialise(ser)
%initialise(ser)
%Opens the COM port and checks it can talk to the robot
%Send the serial port as ser
disp('Calibrating robot...')

out=SendCommand('roboforth', ser);

if isempty(out)  %End initialisation we don't get anything back from the robot
    disp('Error communicating with the robot. Check connection.')
    return
end

SendCommand('start', ser);

%De-energise the robot and bring it to vertical position
SendCommand('de-energize', ser);
input('Manually move robot to a near-vertical position then press enter');

SendCommand('calibrate',ser);

disp('Calibration complete');
