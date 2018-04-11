function [ indices ] = run_experiment( experiment_name, class_name)
%suca
    instrreset

    clear a
    a = arduino;

    ser = serial('COM5', 'Baudrate' , 19200);
    fopen(ser);
    ser_stop=serial('COM4', 'Baudrate', 9600);
    fopen(ser_stop);
    initialise(ser)
    
%     data_matrix = CySkin_Mex;
    
    up_position.point=[0;200;-65];
    up_position.angle=[91.5;0;0];
    
    MoveToCart(up_position,ser);
    
    SetSpeed(1000, ser);

    data=['CARTESIAN ' num2str(up_position.point(1)*10) ' ' num2str(up_position.point(2)*10) ' ' num2str(up_position.point(3)*10-1000) ' MOVETO\r'];    
    fprintf(ser, sprintf(data));
    indices = [];
    i=0;
    stopped = false;
    while true
%         data_matrix = [data_matrix; CySkin_Mex()];
        force = readVoltage(a, 'A0');
        if ~stopped && force>=3.8
            stopped = true;
            fprintf(ser_stop,'%s', 0);
            indices = [indices , i];
        end
        if ~isempty(indices) && size(indices, 2)==1 && i-indices(1)>=20
            indices = [indices, i];
            data=['CARTESIAN ' num2str(up_position.point(1)*10) ' ' num2str(up_position.point(2)*10) ' ' num2str(up_position.point(3)*10) ' MOVETO\r'];    
            fprintf(ser, sprintf(data));
        end
        if ~isempty(indices) && size(indices, 2)==2 && i-indices(2)>=20
            if i-indices(2)>10
                break
            end
        end
        i = i+1;
    end
    
    fclose(ser);
    fprintf(ser_stop,'%s', 0);
    delete(ser);
    clear ser
end

