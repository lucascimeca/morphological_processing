function joints=JointWhere(ser)
%JointWhere(ser) returns where the joints are in the form:
%[WAIST SHOULDER ELBOW L-HAND WRIST]

SendCommand('joint', ser);  %Put in joint mode if not already

joints_str=SendCommand('where', ser);    %Get the string of locations

split=strsplit(joints_str);  %Split the string into an array

joints=[str2num(split{8});str2num(split{9});str2num(split{10});str2num(split{11});str2num(split{12})];

