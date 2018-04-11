function messages_str=SendCommand(command,ser)
%Sends commands to the arm and returns the messages.
%Command need not be in all-caps. Send the serial port as ser.

command_caps=upper(command);         %Convert to all caps if not already

data =sprintf([command_caps '\r']);   %Add a return to the command
fprintf(ser,data);                    %Write command to the serial port

messages=fscanf(ser);                 %read from serial port
messages_str=num2str(messages);       %Convert to string for searching

%warning("");    %Clear warnings

while ~contains(messages_str, 'OK')   %Continue reading the serial until we can find "ok", indicating command is finished
    %pause(0.1)
    messages_str=[messages_str num2str(fscanf(ser))];
    
    if contains(messages_str, 'ABORTED')    %Check we haven't got an error message either
        if contains(messages_str, 'NOT DEFINED')    %If the command isn't defined, chances are the robot needs to be reset
            disp('Recieved error message:')
            disp(messages_str);
            
            disp("Attempting to fix. If this isn't the first time you're seeing this, you're in a loop!")
            
            reinitialise('manual',ser);
            return
        else
            error(messages_str);
            return
        end
    end
    
    %if ~isempty(lastwarn)
    %    disp('Warning messages issued. Reset serial communication')
    %    return
    %end
end