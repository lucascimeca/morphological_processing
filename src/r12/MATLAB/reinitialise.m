function reinitialise(mode, ser)
%Resets the robot, hopefully fixing any issues
%Mode is either 'auto' or 'manual'
%'auto' means it will automatically go back to the home position
%'manual' means the user has to move it back

disp(mode)

if strcmp(mode,'auto')
    GoHome(ser);
end

%disp('Closing serial connection');
%fclose(ser);

%pause(2);

%disp('Reopening serial connection')
%fopen(ser);

disp('Sending ROBOFORTH');
SendCommand('roboforth',ser);

pause(2);
disp('Sending START');
SendCommand('start', ser);

pause(2);
if strcmp(mode,'manual')
    SendCommand('de-energize', ser);
    input('Move robot to near vertical position then press enter')
end

disp('Sending CALIBRATE');
SendCommand('calibrate', ser);
return