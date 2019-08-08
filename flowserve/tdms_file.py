import numpy as np
from nptdms import TdmsFile
from nptdms import tdms
import pandas as pd
import matplotlib.pyplot as plt


#read a tdms file
filenameS = r"C:\Users\krish\Desktop\FlowserveDSExercise/ex1.tdms"
tdms_file = TdmsFile(filenameS)

tdms_groups = tdms_file.groups()
tdms_groups

tdms_Untitled = tdms_file.group_channels("Untitled")

Message_Data_channel_Pump_X = tdms_file.object('Untitled', 'Accel LW236869 Pump X')
Message_Data_data_Pump_X = Message_Data_channel_Pump_X.data

Message_Data_channel_Pump_Y = tdms_file.object('Untitled', 'Accel LW236869 Pump Y')
Message_Data_data_Pump_Y = Message_Data_channel_Pump_Y.data

Message_Data_channel_Pump_Z = tdms_file.object('Untitled', 'Accel LW236869 Pump Z')
Message_Data_data_Pump_Z = Message_Data_channel_Pump_Z.data

Message_Data_channel_Motor_X = tdms_file.object('Untitled', 'Accel LW236867 Motor X')
Message_Data_data_Motor_X = Message_Data_channel_Motor_X.data

Message_Data_channel_Motor_Y = tdms_file.object('Untitled', 'Accel LW236867 Motor Y')
Message_Data_data_Motor_Y = Message_Data_channel_Motor_Y.data

Message_Data_channel_Motor_Z = tdms_file.object('Untitled', 'Accel LW236867 Motor Z')
Message_Data_data_Motor_Z = Message_Data_channel_Motor_Z.data

Message_Data_channel_Temp0 = tdms_file.object('Untitled', 'Temp0')
Message_Data_data_Temp0 = Message_Data_channel_Temp0.data

Message_Data_channel_Temp1 = tdms_file.object('Untitled', 'Temp1')
Message_Data_data_Temp1 = Message_Data_channel_Temp1.data

Message_Data_channel_Suction_Pressure = tdms_file.object('Untitled', 'Suction Pressure')
Message_Data_data_Suction_Pressure = Message_Data_channel_Suction_Pressure.data

Message_Data_channel_Discharge_Pressure = tdms_file.object('Untitled', 'Discharge Pressure')
Message_Data_data_Discharge_Pressure = Message_Data_channel_Discharge_Pressure.data

Message_Data_channel_Flow_Meter = tdms_file.object('Untitled', 'Flow Meter')
Message_Data_data_Flow_Meter = Message_Data_channel_Flow_Meter.data

if len(Message_Data_data_Pump_X) == len(Message_Data_data_Pump_Y) and len(Message_Data_data_Pump_X) == len(Message_Data_data_Pump_Z) :

    data_df_pumps = pd.DataFrame({'Pump X': Message_Data_data_Pump_X,
                            'Pump Y': Message_Data_data_Pump_Y,
                            'Pump Z': Message_Data_data_Pump_Z})
    plt.plot(data_df_pumps['Pump X'], 'r--', data_df_pumps['Pump Y'], 'b--', data_df_pumps['Pump Z'], 'g--')
    plt.savefig('pump graph', format='jpeg')
else:
    print ('the length of the pump data is not equal')

if len(Message_Data_data_Motor_X) == len(Message_Data_data_Motor_Y) and len(Message_Data_data_Motor_X) == len(Message_Data_data_Motor_Z) :

    data_df_motors = pd.DataFrame({'Motor X': Message_Data_data_Motor_X,
                            'Motor Y': Message_Data_data_Motor_Y,
                            'Motor Z': Message_Data_data_Motor_Z})
    plt.plot(data_df_motors['Motor X'], 'r--', data_df_motors['Motor Y'], 'b--', data_df_motors['Motor Z'], 'g--')
    plt.savefig('motor graph', format='jpeg')
else:
    print ('the length of the motor data is not equal')
    plt.plot(Message_Data_data_Motor_X, 'r--', Message_Data_data_Motor_Y, 'b--', Message_Data_data_Motor_Z, 'g--')
    plt.savefig('motor graph', format='jpeg')

if len(Message_Data_data_Temp0) == len(Message_Data_data_Temp1):
    data_df_temp = pd.DataFrame({'Temp 0': Message_Data_data_Temp0,
                            'Temp 1': Message_Data_data_Temp1})
    plt.plot(data_df_temp['Temp 0'], 'r--', data_df_temp['Temp 1'], 'b--')
    plt.savefig('temperature graph', format='jpeg')
else:
    print ('the length of the temperature data is not equal')

if len(Message_Data_data_Suction_Pressure) == len(Message_Data_data_Discharge_Pressure) and len(Message_Data_data_Suction_Pressure) == len(Message_Data_data_Flow_Meter):

    data_df_pressure = pd.DataFrame({'Suction Pressure': Message_Data_data_Suction_Pressure,
                            'Discharge Pressure': Message_Data_data_Discharge_Pressure,
                            'Flow Meter': Message_Data_data_Flow_Meter})
    plt.plot(data_df_pressure['Suction Pressure'], 'r--', data_df_pressure['Discharge Pressure'], 'b--')
    plt.savefig('pressure graph', format='jpeg')
else:
    print ('the length of the pressure data is not equal')

