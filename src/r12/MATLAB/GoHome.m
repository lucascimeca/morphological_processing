function messages=GoHome(ser)
%GoHome(ser)
%Sets speed to default then sends the robotic arm home

SetSpeed(ser);
messages=SendCommand('home', ser);