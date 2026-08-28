# setup applicateion data BPL_IEC
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-28 - Created
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
from pyfmi import load_fmu

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   fmu_model ='BPL_IEC_Column_system_operation_windows_jm_cs.fmu'       
   model = load_fmu(fmu_model, log_level=0)
   flag_vendor = 'JM'
   flag_type = 'CS'
elif platform.system() == 'Linux':  
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['','JM','jm']:    
      print('Linux - run FMU pre-compiled JModelica 2.4')
      fmu_model ='BPL_IEC_Column_system_linux_jm_cs.fmu'      
      model = load_fmu(fmu_model, log_level=0)
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-compiled OpenModelica') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_IEC_Column_system_operation_linux_om_cs.fmu'    
         model = load_fmu(fmu_model, log_level=0)
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_IEC_Column_system_operation_linux_om_me.fmu' 
         model = load_fmu(fmu_model, log_level=0)
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = model.simulate_options()
   opts_std['silent_mode'] = True
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'     
elif flag_type in ['ME', 'me']:
   opts_std = model.simulate_options()
   opts_std["CVode_options"]["verbosity"] = 50 
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'  
else:    
   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   MSL_usage = model.get('MSL.usage')[0]
   MSL_version = model.get('MSL.version')[0]
   BPL_version = model.get('BPL.version')[0]
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '4.1.0 - used components: RealInput, RealOutput, CombiTimeTable, Types' 
   MSL_version = '4.1.0'
   BPL_version = 'Bioprocess Library version 2.3.2' 
else:    
   print('There is no FMU for this platform')
   
#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateValue, parValue, parLocation, parCheck, diagrams, ax, lines
#------------------------------------------------------------------------------------------------------------------
   
# Simulation time
simulationTime = 100.0
prevFinalTime = 0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Create stateValue that later will be used to store final state and used for initialization in 'cont':
stateValue =  {}
stateValue = model.get_states_list()
stateValue.update(timeDiscreteStates)

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = []

# Provide process diagram on disk
fmu_process_diagram ='IBPL_IEC_process_diagram_omnigraffle.png'

# Create dictionaries parValue and parLocation
parValue = {}

parValue['diameter'] = 7.136
parValue['height'] = 20.0
parValue['x_m'] = 0.30
parValue['k1'] = 0.3
parValue['k2'] = 0.05
parValue['k3'] = 0.05
parValue['k4'] = 0.3
parValue['Q_av'] = 3.0

parValue['E_start'] = 0.0

parValue['P_in'] = 0.3
parValue['A_in'] = 0.3
parValue['E_in'] = 0
parValue['E_in_desorption_buffer'] = 0.3

parValue['LFR'] = 0.67

parValue['scale_volume'] = True
parValue['gradient'] = True
parValue['start_adsorption'] = 0
parValue['stop_adsorption'] = 67
parValue['start_desorption'] = 200
parValue['x_start_desorption'] = 0.2
parValue['stationary_desorption'] = 500
parValue['stop_desorption'] = 600
parValue['start_pooling'] = 308
parValue['stop_pooling'] = 600

#parValue['uv_start_trend'] = 0
parValue['start_uv'] = -1
parValue['stop_uv'] = -2

parLocation = {}
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

# Extra and also duplicate names only for describe()     
parLocation['VFR'] = 'F'
parLocation['area'] = 'column.area'
parLocation['V'] = 'column.V'
parLocation['V_m'] = 'column.V_m'

# Parameter value check - especially for hysteresis to avoid runtime error
parCheck = []
parCheck.append("parValue['start_adsorption'] <= parValue['stop_adsorption']")
parCheck.append("parValue['start_desorption'] <= parValue['stationary_desorption']")
parCheck.append("parValue['stationary_desorption'] <= parValue['stop_desorption']")
parCheck.append("parValue['start_uv'] > parValue['stop_uv']")

# Create list of diagrams to be plotted by simu()
diagrams = []

# Create an empty list axes to be defined in newplot() and plotted by simu() or show()
ax = []

# Create list of pens for the diagrams
lines = ['-','--',':','-.']

# Define profile for standard diagrams
def profile(t_n, id):
    data = np.zeros(9)
    data[0] = sim_res['time'][t_n]
    for j in list(range(1,9)):
        data[j] = sim_res['column.column_section[' + str(j) + '].c[' + str(id) + ']'][t_n]
    return data