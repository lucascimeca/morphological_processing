function sensors=GetDataDemo()
%GetDataDemo() fetches the data for the demo sensor array
%Returns a matrix:[taxel1 taxel2 ... taxel10    0   ;   <- sensor 1
%                  taxel1 taxel2 ... taxel10    0   ;   <- sensor 2
%                      :    :     :     :       :   ;
%                  taxel1 taxel2 ... taxel10 taxel11]   <- sensor 6


raw=CySkin_Mex;     %Reads the data at this instant, returning a 1x69 matrix


sensors=[];     %Initialise the matrix

sensors(1,:)=[raw(1,4:13) 0];   %Extract entries 4-13 from the raw data, corresponding to sensor 1 and add a
                                %0 entry as there's only 10 elements
sensors(2,:)=[raw(1,15:24) 0];  %Extract entries 15-24 from the raw data, corresponding to sensor 2 and add a
                                %0 entry as there's only 10 elements
sensors(3,:)=[raw(1,38:47) 0];  %Extract entries 38-47 from the raw data, corresponding to sensor 4 and add a
                                %0 entry as there's only 10 elements
sensors(4,:)=[raw(1,49:58) 0];  %Extract entries 49-58 from the raw data, corresponding to sensor 5 and add a
                                %0 entry as there's only 10 elements
sensors(5,:)=[raw(1,60:69) 0];  %Extract entries  60-69 from the raw data, corresponding to sensor 6 and add a
                                %0 entry as there's only 10 elements                                

sensors(6,:)=raw(1,26:36);      %Extract entries 15-24 from the raw data, corresponding to sensor 3 and don't
                                %add the 0 entry, as this is the one with 11 taxels