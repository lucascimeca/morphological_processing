function position=CartWhere(ser)
%CartWhere(ser) returns where the end-effector is in the form:
%position.point = [X;Y;Z]
%position.angle = [PITCH;ROLL;LEN]

SendCommand('cartesian', ser);  %Put in cartesian mode if not already

point_str=SendCommand('where', ser);    %Get the string of locations

split=strsplit(point_str);  %Split the string into an array

position.point=[str2num(split{9});str2num(split{10});str2num(split{11})];

position.angle=[str2num(split{12});str2num(split{13});str2num(split{14})];
