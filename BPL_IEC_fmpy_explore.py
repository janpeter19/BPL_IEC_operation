# Figure - Simulation of IEC
#          with functions added to facilitate explorative simulation work
#
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2021-09-24 - Created
# 2021-09-27 - Include also elution phase
# 2021-09-29 - Structured column system slightly differently and updated for that
# 2021-10-01 - Updated system_info() with FMU-explore version
# 2021-10-01 - Updated diagrams and used uv_detector signal 
# 2021-11-12 - After talking with Karl Johan Brink I changed volume to area*height and scale by area
# 2021-11-13 - After talking with Karl Johan I also introduce linear flow rate u i.e. u = F/area
# 2021-11-13 - Also changed time unit from s to min all over and according to Karl Johan
# 2021-11-25 - Introduced diagram Elution-vs-volume and Elution-vs-volume-combined
# 2021-11-27 - Modifed for F, u, V 
# 2021-12-02 - Update for use of FluidMixerV in BPL ver 2.0.9 - beta 
# 2021-12-03 - Added simple plot of column outlet vs time to bridge to OpenModelica demo
# 2021-12-03 - Extended FMU-explore 0.8.6 for function disp() and dictionary parLocation[]
# 2021-12-13 - Extended newplot() with diagrams inclukding concductivity instead of ion concentration
# 2021-12-13 - Change unit from min to hours and also affect process parameters
# 2021-12-14 - Changed start and point for diagrams related to elution to be set automatically based on parDict
# 2021-12-17 - Now adjusted diagrams Elution-pooling
# 2021-12-18 - Changed back to use unit mL for all volumes and what Karl Johan wanted
# 2021-12-18 - Correction of disp() - now FMU-explore ver 0.8.7
# 2021-12-22 - Change of how flows are controlled and their parameters
# 2022-04-25 - Update to FMU-explore 0.9.0.
# 2022-04-25 - Take away variable scaling and read it off from mode() when needed
# 2022-04-25 - Introduce a switch called scale_volume that is true for using volume for switch events alt time
# 2022-04-27 - Modified disp() and describe() to handle that scale_volume is boolean
# 2022-04-28 - Tidy up newplot()
# 2022-04-29 - Corrected newplot()  'Elution-pooling' and 'Pooling', also improved error text par() and init() 
# 2022-05-01 - Corrected newplot() 'Time' to 'time' and import when Jupyter widget framework is used
# 2022-05-07 - Changed time scale from hours to min
# 2022-05-21 - Added to newplot 'Elution-conductivity-combined-all'
# 2022-10-07 - Updated for FMU-explore 0.9.5 with disp() that do not include extra parameters with parLocation
# 2022-11-11 - Updating handling of V, V_m and LFR
# 2022-11-12 - Updating newplot() with plotType 'Elution-vs-CV
# 2022-11-14 - Introduced possibilty to give intial value of ion E in section 1 a value to illustrate need to wash
# 2022-12-01 - Added parameters for control_pooling based on UV-levels in combination with a time_window
# 2022-12-03 - Introduced diagram plotType='Elutions-vs-CV-pooling'
# 2022-12-09 - Adjusted to skip pooling2 for a while
# 2022-12-12 - FNU-explore 0.9.6b test with extension to par() using dictionary parCheck
# 2022-12-12 - Changed uv_high and uv_low to start_uv and stop_uv and also changed parLocation after FMU update
# 2022-12-16 - Updated describe() test for 'chromatogoraph' as well as for 'liquidphase' for clarity
# 2023-01-28 - Include E_in_sample an idea from Karl Johan Brink
# 2023-02-03 - Include a switch gradient that if true produce a salt gradient and if false stepwise increase
# 2023-02-06 - Change to ControlDesrptionBuffer and corresponding changes in parDict and parLocation etc
# 2023-02-06 - Included relevant list of parCheck
# 2023-02-06 - Updated to FMU-explore 0.9.6e including parCheck...
# 2023-02-06 - Play with the idea of parCalc - but dropped
# 2023-04-05 - Update FMU-explore 0.9.7
# 2023-04-17 - Modify for FMU-explore 0.9.7 for FMPy
# 2023-04-18 - Modify for FMU-explore 0.9.8 for FMPy - update model_get() for Boolean variables
# 2023-05-31 - Adjusted to from importlib.meetadata import version
# 2023-06-02 - Add logging of a few variables
# 2023-09-14 - Update FMU-explore 0.9.8 (try to keep the same ver number as for PyFMI) with process diagram
# 2024-03-09 - Update FMU-explore 0.9.9
# 2024-05-14 - Polish the script
# 2024-05-20 - Updated the OpenModelica version to 1.23.0-dev
# 2024-06-01 - Corrected model_get() to handle string values as well - improvement very small and keep ver 1.0.0
# 2024-10-04 - Update information about FMU and change NCP to ncp
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
import zipfile  

from fmpy import simulate_fmu
from fmpy import read_model_description
import fmpy as fmpy

from itertools import cycle
from importlib.metadata import version 

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
global fmu_model
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   fmu_model ='BPL_IEC_Column_system_windows_jm_cs.fmu'       
   model_description = read_model_description(fmu_model)  
   flag_vendor = 'JM'
   flag_type = 'CS'
elif platform.system() == 'Linux': 
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['','JM','jm']:    
      print('Linux - run FMU pre-compiled JModelica 2.4')
      fmu_model ='BPL_IEC_Column_system_linux_jm_cs.fmu'      
      model_description = read_model_description(fmu_model) 
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-compiled OpenModelica') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_IEC_Column_system_linux_om_cs.fmu'    
         model_description = read_model_description(fmu_model) 
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_IEC_Column_system_linux_om_me.fmu' 
         model_description = read_model_description(fmu_model) 
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = {'ncp': 500}
elif flag_type in ['ME', 'me']:
   opts_std = {'ncp': 500}
else:    
   print('There is no FMU for this platform')

# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   constants = [v for v in model_description.modelVariables if v.causality == 'local'] 
   MSL_usage = [x[1] for x in [(constants[k].name, constants[k].start) \
                     for k in range(len(constants))] if 'MSL.usage' in x[0]][0]   
   MSL_version = [x[1] for x in [(constants[k].name, constants[k].start) \
                       for k in range(len(constants))] if 'MSL.version' in x[0]][0]
   BPL_version = [x[1] for x in [(constants[k].name, constants[k].start) \
                       for k in range(len(constants))] if 'BPL.version' in x[0]][0] 
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '3.2.3 - used components: RealInput, RealOutput, CombiTimeTable, Types' 
   MSL_version = '3.2.3'
   BPL_version = 'Bioprocess Library version 2.2.1 - GUI' 
else:    
   print('There is no FMU for this platform')

# Simulation time
global simulationTime; simulationTime = 100.0
global prevFinalTime; prevFinalTime = 0


# Dictionary of time discrete states
timeDiscreteStates = {} 

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = []

# Provide process diagram on disk
fmu_process_diagram ='IBPL_IEC_process_diagram_omnigraffle.png'

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateDict, parDict, diagrams, newplot(), describe()
#------------------------------------------------------------------------------------------------------------------
   
# Create stateDict that later will be used to store final state and used for initialization in 'cont':
global stateDict; stateDict =  {}
stateDict = {variable.derivative.name:None for variable in model_description.modelVariables \
                                            if variable.derivative is not None}
stateDict.update(timeDiscreteStates) 

global stateDictInitial; stateDictInitial = {}
for key in stateDict.keys():
    if not key[-1] == ']':
         if key[-3:] == 'I.y':
            stateDictInitial[key] = key[:-10]+'I_start'
         elif key[-3:] == 'D.x':
            stateDictInitial[key] = key[:-10]+'D_start'
         else:
            stateDictInitial[key] = key+'_start'
    elif key[-3] == '[':
        stateDictInitial[key] = key[:-3]+'_start'+key[-3:]
    elif key[-4] == '[':
        stateDictInitial[key] = key[:-4]+'_start'+key[-4:]
    elif key[-5] == '[':
        stateDictInitial[key] = key[:-5]+'_start'+key[-5:] 
    else:
        print('The state vector has more than 1000 states')
        break

global stateDictInitialLoc; stateDictInitialLoc = {}
for value in stateDictInitial.values():
    stateDictInitialLoc[value] = value

# Create dictionaries parDict and parLocation
global parDict; parDict = {}

parDict['diameter'] = 7.136
parDict['height'] = 20.0
parDict['x_m'] = 0.30
parDict['k1'] = 0.3
parDict['k2'] = 0.05
parDict['k3'] = 0.05
parDict['k4'] = 0.3
parDict['Q_av'] = 3.0

parDict['E_start'] = 0.0

parDict['P_in'] = 0.3
parDict['A_in'] = 0.3
parDict['E_in'] = 0
parDict['E_in_desorption_buffer'] = 0.3

parDict['LFR'] = 0.67

parDict['scale_volume'] = True
parDict['gradient'] = True
parDict['start_adsorption'] = 0
parDict['stop_adsorption'] = 67
parDict['start_desorption'] = 200
parDict['x_start_desorption'] = 0.2
parDict['stationary_desorption'] = 500
parDict['stop_desorption'] = 600
parDict['start_pooling'] = 308
parDict['stop_pooling'] = 600

#parDict['uv_start_trend'] = 0
parDict['start_uv'] = -1
parDict['stop_uv'] = -2

global parLocation; parLocation = {}
parLocation['diameter'] = 'column.diameter'
parLocation['height'] = 'column.height'
parLocation['x_m'] = 'column.x_m'
parLocation['k1'] = 'column.k1'
parLocation['k2'] = 'column.k2'
parLocation['k3'] = 'column.k3'
parLocation['k4'] = 'column.k4'
parLocation['Q_av'] = 'column.Q_av'

parLocation['E_start'] = 'column.column_section[1].c_start[3]'

parLocation['P_in'] = 'tank_sample.c_in[1]'
parLocation['A_in'] = 'tank_sample.c_in[2]'
parLocation['E_in'] = 'tank_sample.c_in[3]'
parLocation['E_in_desorption_buffer'] = 'tank_buffer2.c_in[3]'

parLocation['LFR'] = 'u'

parLocation['scale_volume'] = 'scale_volume'
parLocation['gradient'] = 'control_desorption_buffer.gradient'
parLocation['start_adsorption'] = 'control_sample.start'
parLocation['stop_adsorption'] = 'control_sample.stop'
parLocation['start_desorption'] = 'control_desorption_buffer.start'
parLocation['x_start_desorption'] = 'control_desorption_buffer.x_start'
parLocation['stationary_desorption'] = 'control_desorption_buffer.stationary'
parLocation['stop_desorption'] = 'control_desorption_buffer.stop'
parLocation['start_pooling'] = 'control_pooling.start'
parLocation['stop_pooling'] = 'control_pooling.stop'

#parLocation['uv_start_trend'] = 'control_pooling2.uv_start_trend'
parLocation['start_uv'] = 'control_pooling.start_uv_pooling'
parLocation['stop_uv'] = 'control_pooling.stop_uv_pooling'

# Extra only for describe()
global key_variables; key_variables = []
parLocation['V'] = 'column.V'; key_variables.append(parLocation['V'])
parLocation['scale_volume'] = 'scale_volume'; key_variables.append(parLocation['scale_volume'])
parLocation['VFR'] = 'F'; key_variables.append(parLocation['VFR'])
parLocation['area'] = 'column.area'; key_variables.append(parLocation['area'])
parLocation['V_m'] = 'column.V_m'; key_variables.append(parLocation['V_m'])

parLocation['column.column_section[1].V_m'] = 'column.column_section[1].V_m'; 
key_variables.append(parLocation['column.column_section[1].V_m'])

parLocation['tank_mixing.outlet.c[1]'] ='tank_mixing.outlet.c[1]'; 
key_variables.append(parLocation['tank_mixing.outlet.c[1]'])

parLocation['control_buffer2.scaling'] ='control_buffer2.scaling'; 
key_variables.append(parLocation['control_buffer2.scaling'])

# Parameter value check - especially for hysteresis to avoid runtime error
global parCheck; parCheck = []
parCheck.append("parDict['start_adsorption'] < parDict['stop_adsorption']")
parCheck.append("parDict['start_desorption'] < parDict['stationary_desorption']")
parCheck.append("parDict['stationary_desorption'] < parDict['stop_desorption']")
parCheck.append("parDict['start_uv'] > parDict['stop_uv']")


# Create list of diagrams to be plotted by simu()
global diagrams
diagrams = []

# Define standard plots
def profile(t_n, id):
    data = np.zeros(9)
    data[0] = sim_res['time'][t_n]
    for j in list(range(1,9)):
        data[j] = sim_res['column.column_section[' + str(j) + '].c[' + str(id) + ']'][t_n]
    return data

def newplot(title='IEC', plotType='Loading'):
   """ Standard plot window 
       title = '' """
   
   # Globals
   global ax1, ax2, ax3, ax4, ax5, ax6    
   global ax11, ax12, ax21, ax22
    
   # Reset pens
   setLines()

   # Plot diagram 
   if plotType == 'Loading':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[PS] and c[AS][mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[PS] and c[AS][mg/mL]')
      ax2.set_xlabel('Sections in column - inlet to outlet') 
      
      # Part of plot made after simulation
      diagrams.clear()
      diagrams.append("ax1.plot(list(range(1,9)), profile(10,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(list(range(1,9)), profile(50,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(150,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(200,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(250,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(300,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(350,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(400,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(450,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(500,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(10,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(50,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(150,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(200,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(250,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(300,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(350,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(400,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(450,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(500,5)[1:], 'r')")
      diagrams.append("ax2.plot(list(range(1,9)), profile(500,4)[1:], 'b*-')")      
      diagrams.append("ax2.plot(list(range(1,9)), profile(500,5)[1:], 'r*-')")      
        
   elif plotType == 'Loading-combined':
      
      # Part of plot made before simulation   
      plt.figure()
      ax11 = plt.subplot(2,2,1)
      ax12 = plt.subplot(2,2,2)
      ax21 = plt.subplot(2,2,3)
      ax22 = plt.subplot(2,2,4)

      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('c[P] and c[A][mg/mL]')

      ax12.grid()
      ax12.set_ylabel('c[PS] and c[AS][mg/mL]')
           
      ax21.grid()
      ax21.set_ylabel('Tank_waste [mL]')
      ax21.set_xlabel('Time [min]')
   
      ax22.grid()
      ax22.set_ylabel('c[PS] and c[AS][mg/mL]')       
      ax22.set_xlabel('Section in column - inlet to outlet') 

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax11.plot(sim_res['time'], sim_res['tank_mixing.outlet.c[1]'], color='b', linestyle=linetype)")           
      diagrams.append("ax12.plot(list(range(1,9)), profile(10,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(50,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(150,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(200,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(250,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(300,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(350,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(400,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(450,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(500,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(10,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(50,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(150,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(200,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(250,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(300,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(350,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(400,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(450,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(500,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax21.plot(sim_res['time'], sim_res['tank_waste.V'], color='b', linestyle=linetype)")
      diagrams.append("ax22.plot(list(range(1,9)), profile(500,4)[1:], color='b', linestyle=linetype)")      
      diagrams.append("ax22.plot(list(range(1,9)), profile(500,5)[1:], color='r', linestyle=linetype)")  
      
   elif plotType == 'Elution':
      
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Time [min] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
   elif plotType == 'Elution-vs-volume':
         
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Pumped liquid volume [mL] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")

   elif plotType == 'Elution-vs-CV':
         
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Pumped liquid volume [CV] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")

   elif plotType == 'Elution-vs-volume-all':
         
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Pumped liquid volume [mL]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'], \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF'], \
                                0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")


   elif plotType == 'Elution-conductivity-vs-volume':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('UV-detector []')
 
      ax3.grid()
      ax3.set_ylabel('Conductivity [mS/cm]')      
      ax3.set_xlabel('Pumped liquid volume [mL]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")

      diagrams.append("ax3.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)") 

   elif plotType == 'Elution-conductivity-vs-volume-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('UV-detector []')
 
      ax3.grid()
      ax3.set_ylabel('Conductivity [mS/cm]')      
      ax3.set_xlabel('Pumped liquid volume [mL]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'], \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 

      diagrams.append("ax3.plot(sim_res['ackF'], \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)") 

   elif plotType == 'Elution-combined':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(8,1,5)
      ax3 = plt.subplot(8,1,6)
      ax4 = plt.subplot(8,1,7)
      ax5 = plt.subplot(8,1,8)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A], c[E] [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('F sample [mL/min]')

      ax3.grid()
      ax3.set_ylabel('F buff1 [mL/min]')

      ax4.grid()
      ax4.set_ylabel('F buff2 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('V prod [L]')
      ax5.set_xlabel('Time [min]')  

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'], 0.05*sim_res['column.column_section[8].outlet.c[3]'], label='E', color='m', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.step(sim_res['time'], sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax3.plot(sim_res['time'], sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax4.plot(sim_res['time'], sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax5.step(sim_res['time'], sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 
  
   elif plotType == 'Elution-vs-volume-combined':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(8,1,5)
      ax3 = plt.subplot(8,1,6)
      ax4 = plt.subplot(8,1,7)
      ax5 = plt.subplot(8,1,8)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A], c[E] [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('F sample')

      ax3.grid()
      ax3.set_ylabel('F buffer 1')

      ax4.grid()
      ax4.set_ylabel('F buffer 2')

      ax5.grid()
      ax5.set_ylabel('V harvest [mL]')
      ax5.set_xlabel('Pumped liquid volume [mL]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='E', color='m', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax3.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax4.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax5.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 

   elif plotType == 'Elution-conductivity-vs-volume-combined':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
 
      #ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Pumped liquid volume [mL]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax1.set_ylim([0, 1.05*max(sim_res['column.column_section[8].outlet.c[1]'])])")
      
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 
      diagrams.append("ax1.set_xlim(0)")
      diagrams.append("ax2.set_xlim(0)")
      diagrams.append("ax3.set_xlim(0)")
      diagrams.append("ax4.set_xlim(0)")
      diagrams.append("ax5.set_xlim(0)")
      diagrams.append("ax6.set_xlim(0)")

   elif plotType == 'Elution-conductivity-vs-volume-combined-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
 
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Pumped liquid volume [mL]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'], sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'], sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax2.plot(sim_res['ackF'], sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.step(sim_res['ackF'], sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF'], sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF'], sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF'], sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 

   elif plotType == 'Elution-conductivity-vs-CV-combined-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
      
      #ax2 = plt.subplot(8,1,4)
      #ax3 = plt.subplot(8,1,5)
      #ax4 = plt.subplot(8,1,6)
      #ax5 = plt.subplot(8,1,7)
      #ax6 = plt.subplot(8,1,8)
 
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Pumped liquid volume [CV]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax2.plot(sim_res['ackF']/model_get('column.V'), sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.step(sim_res['ackF']/model_get('column.V'), sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF']/model_get('column.V'), sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF']/model_get('column.V'), sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF']/model_get('column.V'), sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 


   elif plotType == 'Elution-conductivity-combined-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
 
      #ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Time [min] - relative start desorption')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.step(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 

   elif plotType == 'Elution-pooling':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(6,1,5)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      
      ax3.grid()
      ax3.set_ylabel('Pooling [0/1]')
      ax3.set_xlabel('Time [min]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
      diagrams.append("ax3.step(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['control_pooling.out'], color='k', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)")      

   elif plotType == 'Elution-vs-CV-pooling':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(6,1,5)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A], c[E]  [mg/mL]')
      
      ax3.grid()
      ax3.set_ylabel('Pooling [0/1]')
      ax3.set_xlabel('Pumped liquid volume [CV]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
     # diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF']/model_get('column.V'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF']/model_get('column.V'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
    # diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
      diagrams.append("ax3.step(sim_res['ackF']/model_get('column.V'), \
                                sim_res['control_pooling.out'], color='k', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)")      


   elif plotType == 'Pooling':
      
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('m[P], m[A] - harvest  [mg]')
          
      ax2.grid()
      ax2.set_ylabel('m[P], m[A] - waste  [mg]')
    
      ax3.grid()
      ax3.set_ylabel('Pooling [0/1]')
      ax3.set_xlabel('Time [min]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'], sim_res['tank_harvest.m[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'], sim_res['tank_harvest.m[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")

      diagrams.append("ax2.plot(sim_res['time'], sim_res['tank_waste.m[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'], sim_res['tank_waste.m[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax2.legend()")
       
      diagrams.append("ax3.step(sim_res['time'], sim_res['control_pooling.out'], color='k', linestyle=linetype)")

   elif plotType == 'Column-outlet':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P]')
          
      ax2.grid()
      ax2.set_ylabel('c[A]')
    
      ax3.grid()
      ax3.set_ylabel('c[E]')
      ax3.set_xlabel('Time [min]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.out\x6C\x65\x74\x2E\x63\x5B\x31\x5D\x27\x5D\x2C\x20\x6C\x61\x62\x65\x6C\x3D\x27\x50\x27\x2C\x20\x63\x6F\x6C\x6F\x72\x3D\x27\x62\x27\x2C\x20\x6C\x69\x6E\x65\x73\x74\x79\x6C\x65\x3D\x6C\x69\x6E\x65\x74\x79\x70\x65\x29\x22\x29\x0D\x0A\x20\x20\x20\x20\x20\x20\x64\x69\x61\x67\x72\x61\x6D\x73\x2E\x61\x70\x70\x65\x6E\x64\x28\x22\x61\x78\x32\x2E\x70\x6C\x6F\x74\x28\x73\x69\x6D\x5F\x72\x65\x73\x5B\x27\x74\x69\x6D\x65\x27\x5D\x2C\x20\x73\x69\x6D\x5F\x72\x65\x73\x5B\x27\x63\x6F\x6C\x75\x6D\x6E\x2E\x6F\x75\x74\x6C\x65\x74\x2E\x63\x5B\x32\x5D\x27\x5D\x2C\x20\x6C\x61\x62\x65\x6C\x3D\x27\x50\x27\x2C\x20\x63\x6F\x6C\x6F\x72\x3D\x27\x62\x27\x2C\x20\x6C\x69\x6E\x65\x73\x74\x79\x6C\x65\x3D\x6C\x69\x6E\x65\x74\x79\x70\x65\x29\x22\x29\x0D\x0A\x20\x20\x20\x20\x20\x20\x64\x69\x61\x67\x72\x61\x6D\x73\x2E\x61\x70\x70\x65\x6E\x64\x28\x22\x61\x78\x33\x2E\x70\x6C\x6F\x74\x28\x73\x69\x6D\x5F\x72\x65\x73\x5B\x27\x74\x69\x6D\x65\x27\x5D\x2C\x20\x73\x69\x6D\x5F\x72\x65\x73\x5B\x27\x63\x6F\x6C\x75\x6D\x6E\x2E\x6F\x75\x74\x6C\x65\x74\x2E\x63\x5B\x33\x5D\x27\x5D\x2C\x20\x6C\x61\x62\x65\x6C\x3D\x27\x41\x27\x2C\x20\x63\x6F\x6C\x6F\x72\x3D\x27\x72\x27\x2C\x20\x6C\x69\x6E\x65\x73\x74\x79\x6C\x65\x3D\x6C\x69\x6E\x65\x74\x79\x70\x65\x29\x22\x29\x0D\x0A\x0D\x0A\x20\x20\x20\x65\x6C\x73\x65\x3A\x0D\x0A\x20\x20\x20\x20\x20\x20\x70\x72\x69\x6E\x74\x28\x22\x50\x6C\x6F\x74\x20\x77\x69\x6E\x64\x6F\x77\x20\x74\x79\x70\x65\x20\x6E\x6F\x74\x20\x63\x6F\x72\x72\x65\x63\x74\x22\x29\x20\x0D\x0A\x0D\x0A\x23\x20\x44\x65\x66\x69\x6E\x65\x20\x61\x6E\x64\x20\x65\x78\x74\x65\x6E\x64\x20\x64\x65\x73\x63\x72\x69\x62\x65\x20\x66\x6F\x72\x20\x74\x68\x65\x20\x63\x75\x72\x72\x65\x6E\x74\x20\x61\x70\x70\x6C\x69\x63\x61\x74\x69\x6F\x6E\x0D\x0A\x64\x65\x66\x20\x64\x65\x73\x63\x72\x69\x62\x65\x28\x6E\x61\x6D\x65\x2C\x20\x64\x65\x63\x69\x6D\x61\x6C\x73\x3D\x33\x29\x3A\x0D\x0A\x20\x20\x20\x22\x22\x22\x4C\x6F\x6F\x6B\x20\x75\x70\x20\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6F\x6E\x20\x6F\x66\x20\x63\x75\x6C\x74\x75\x72\x65\x2C\x20\x6D\x65\x64\x69\x61\x2C\x20\x61\x73\x20\x77\x65\x6C\x6C\x20\x61\x73\x20\x70\x61\x72\x61\x6D\x65\x74\x65\x72\x73\x20\x61\x6E\x64\x20\x76\x61\x72\x69\x61\x62\x6C\x65\x73\x20\x69\x6E\x20\x74\x68\x65\x20\x6D\x6F\x64\x65\x6C\x20\x63\x6F\x64\x65\x22\x22\x22\x0D\x0A\x0D\x0A\x20\x20\x20\x69\x66\x20\x6E\x61\x6D\x65\x20\x3D\x3D\x20\x27\x63\x68\x72\x6F\x6D\x61\x74\x6F\x67\x72\x61\x70\x68\x79\x27\x3A\x0D\x0A\x20\x20\x20\x20\x20\x20\x70\x72\x69\x6E\x74\x28\x27\x49\x6F\x6E\x20\x65\x78\x63\x68\x61\x6E\x67\x65\x20\x63\x68\x72\x6F\x6D\x61\x74\x6F\x72\x67\x72\x61\x70\x68\x79\x20\x63\x6F\x6E\x74\x72\x6F\x6C\x6C\x65\x64\x20\x77\x69\x74\x68\x20\x76\x61\x72\x79\x69\x6E\x67\x20\x73\x61\x6C\x74\x2D\x63\x6F\x6E\x63\x65\x6E\x74\x72\x61\x74\x69\x6F\x6E\x2E\x20\x54\x68\x65\x20\x70\x48\x20\x69\x73\x20\x6B\x65\x70\x74\x20\x63\x6F\x6E\x73\x74\x61\x6E\x74\x2E\x27\x29\x20\x20\x20\x20\x20\x20\x20\x20\x0D\x0A\x0D\x0A\x20\x20\x20\x65\x6C\x69\x66\x20\x6E\x61\x6D\x65\x20\x69\x6E\x20\x5B\x27\x6C\x69\x71\x75\x69\x64\x70\x68\x61\x73\x65\x27\x2C\x20\x27\x6D\x65\x64\x69\x61\x27\x5D\x3A\x0D\x0A\x20\x20\x20\x20\x20\x20\x50\x20\x3D\x20\x6D\x6F\x64\x65\x6C\x5F\x67\x65\x74\x28\x27\x6C\x69\x71\x75\x69\x64\x70\x68\x61\x73\x65\x2E\x50\x27\x29\x3B\x20\x50\x5F\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6F\x6E\x20\x3D\x20\x6D\x6F\x64\x65\x6C\x5F\x67\x65\x74\x5F\x76\x61\x72\x69\x61\x62\x6C\x65\x5F\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6F\x6E\x28\x27\x6C\x69\x71\x75\x69\x64\x70\x68\x61\x73\x65\x2E\x50\x27\x29\x3B\x20\x0D\x0A\x20\x20\x20\x20\x20\x20\x50\x5F\x6D\x77\x20\x3D\x20\x6D\x6F\x64\x65\x6C\x5F\x67\x65\x74\x28\x27\x6C\x69\x71\x75\x69\x64\x70\x68\x61\x73\x65\x2E\x6D\x77\x5B\x31\x5D\x27\x29\x0D\x0A\x20\x20\x20\x20\x20\x20\x41\x20\x3D\x20\x6D\x6F\x64\x65\x6C\x5F\x67\x65\x74\x28\x27\x6C\x69\x71\x75\x69\x64\x70\x68\x61\x73\x65\x2E\x41\x27\x29\x3B\x20\x41\x5F\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6F\x6E\x20\x3D\x20\x6D\x6F\x64\x65\x6C\x5F\x67\x65\x74\x5F\x76\x61\x72\x69\x61\x62\x6C\x65\x5F\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6F\x6E\x28\x27\x6C\x69\x71\x75\x69\x64\x70\x68\x61\x73\x65\x2E\x41\x27\x29\x3B\x20\x0D\x0A\x20\x20\x20\x20\x20\x2 