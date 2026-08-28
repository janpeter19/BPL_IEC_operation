# setup application functions BPL_IEC_operation dependent on previous import of functions from fmu_explore 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-28 - Created
#------------------------------------------------------------------------------------------------------------------

def newplot(title='IEC', plotType='Loading'):
   """ Standard plot window 
       title = '' """
       
   # Reset pens
   resetPen()

   # Plot diagram 
   if plotType == 'Loading':
         
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
    
      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('c[PS] and c[AS][mg/mL]')
    
      ax[1].grid()
      ax[1].set_ylabel('c[PS] and c[AS][mg/mL]')
      ax[1].set_xlabel('Sections in column - inlet to outlet') 
      
      # Part of plot made after simulation
      diagrams.clear()
      diagrams.append("ax[0].plot(list(range(1,9)), profile(10,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(50,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(150,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(200,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(250,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(300,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(350,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(400,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(450,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(500,4)[1:], 'b')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(10,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(50,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(150,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(200,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(250,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(300,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(350,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(400,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(450,5)[1:], 'r')")
      diagrams.append("ax[0].plot(list(range(1,9)), profile(500,5)[1:], 'r')")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(500,4)[1:], 'b*-')")      
      diagrams.append("ax[1].plot(list(range(1,9)), profile(500,5)[1:], 'r*-')")   
        
   elif plotType == 'Loading-combined':
      
      ax11 = plt.subplot(2,2,1)
      ax12 = plt.subplot(2,2,2)
      ax21 = plt.subplot(2,2,3)
      ax22 = plt.subplot(2,2,4)

      ax.clear()
      ax.append(ax11)
      ax.append(ax12)
      ax.append(ax21)
      ax.append(ax22)

      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('c[P] and c[A][mg/mL]')

      ax[1].grid()
      ax[1].set_ylabel('c[PS] and c[AS][mg/mL]')
           
      ax[2].grid()
      ax[2].set_ylabel('Tank_waste [mL]')
      ax[2].set_xlabel('Time [min]')
   
      ax[3].grid()
      ax[3].set_ylabel('c[PS] and c[AS][mg/mL]')       
      ax[3].set_xlabel('Section in column - inlet to outlet') 

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax[0].plot(sim_res['time'], sim_res['tank_mixing.outlet.c[1]'], color='b', linestyle=linetype)")           
      diagrams.append("ax[1].plot(list(range(1,9)), profile(10,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(50,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(150,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(200,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(250,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(300,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(350,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(400,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(450,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(500,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(10,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(50,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(150,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(200,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(250,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(300,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(350,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(400,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(450,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[1].plot(list(range(1,9)), profile(500,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax[2].plot(sim_res['time'], sim_res['tank_waste.V'], color='b', linestyle=linetype)")
      diagrams.append("ax[3].plot(list(range(1,9)), profile(500,4)[1:], color='b', linestyle=linetype)")      
      diagrams.append("ax[3].plot(list(range(1,9)), profile(500,5)[1:], color='r', linestyle=linetype)")  
      
      
   elif plotType == 'Elution':
      
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)

      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
    
      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax[1].grid()
      ax[1].set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax[1].set_xlabel('Time [min] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax[0].plot(sim_res['time']-parValue['start_desorption']/model.get('control_desorption_buffer.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax[0].plot(sim_res['time']-parValue['start_desorption']/model.get('control_desorption_buffer.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax[0].set_xlim(left=0)")
      diagrams.append("ax[0].set_ylim([0,0.45])")
      diagrams.append("ax[0].legend()")
 
      diagrams.append("ax[1].plot(sim_res['time']-parValue['start_desorption']/model.get('control_desorption_buffer.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax[1].plot(sim_res['time']-parValue['start_desorption']/model.get('control_desorption_buffer.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax[1].set_xlim(left=0)") 
      diagrams.append("ax[1].set_ylim([0,0.45])")
      diagrams.append("ax[1].legend()")

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
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
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
      diagrams.append("ax1.plot((sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'))/model.get('column.V')[0], \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot((sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'))/model.get('column.V')[0], \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot((sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'))/model.get('column.V')[0], \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot((sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'))/model.get('column.V')[0], \
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
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")

      diagrams.append("ax3.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
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
         
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(8,1,5)
      ax3 = plt.subplot(8,1,6)
      ax4 = plt.subplot(8,1,7)
      ax5 = plt.subplot(8,1,8)
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
      ax.append(ax3)
      ax.append(ax4)
      ax.append(ax5)
    
      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('c[P], c[A], c[E] [mg/mL]')
    
      ax[1].grid()
      ax[1].set_ylabel('F sample [mL/min]')

      ax[2].grid()
      ax[2].set_ylabel('F buff1 [mL/min]')

      ax[3].grid()
      ax[3].set_ylabel('F buff2 [mL/min]')

      ax[4].grid()
      ax[4].set_ylabel('V prod [L]')
      ax[4].set_xlabel('Time [min]')  

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax[0].plot(sim_res['time'], sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax[0].plot(sim_res['time'], sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax[0].plot(sim_res['time'], 0.05*sim_res['column.column_section[8].outlet.c[3]'], label='E', color='m', linestyle=linetype)")
      diagrams.append("ax[0].legend()")
      
      diagrams.append("ax[1].step(sim_res['time'], sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax[2].plot(sim_res['time'], sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax[3].plot(sim_res['time'], sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax[4].step(sim_res['time'], sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 
  
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
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='E', color='m', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax3.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax4.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax5.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
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
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax1.set_ylim([0, 1.05*max(sim_res['column.column_section[8].outlet.c[1]'])])")
      
      diagrams.append("ax2.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
                                sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF'] - parValue['start_desorption']*model.get('F')/model.get('control_buffer2.scaling'), \
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
         
      ax1 = plt.subplot(2,1,1)    
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
      ax.append(ax3)
      ax.append(ax4)
      ax.append(ax5)
      ax.append(ax6)

      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('c[P], c[A] [mg/mL]')

      ax[1].grid()
      ax[1].set_ylabel('c[E]')      

      ax[2].grid()
      ax[2].set_ylabel('F_sample')

      ax[3].grid()
      ax[3].set_ylabel('Fb1')

      ax[4].grid()
      ax[4].set_ylabel('Fb2')

      ax[5].grid()
      ax[5].set_ylabel('V_pool')
      ax[5].set_xlabel('Pumped liquid volume [CV]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax[0].plot(sim_res['ackF']/model.get('column.V')[0], sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax[0].plot(sim_res['ackF']/model.get('column.V')[0], sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax[0].legend()")
      diagrams.append("ax[1].plot(sim_res['ackF']/model.get('column.V')[0], sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax[2].step(sim_res['ackF']/model.get('column.V')[0], sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax[3].plot(sim_res['ackF']/model.get('column.V')[0], sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax[4].plot(sim_res['ackF']/model.get('column.V')[0], sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax[5].plot(sim_res['ackF']/model.get('column.V')[0], sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 


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

      ax[2].grid()
      ax[2].set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Time [min] - relative start desorption')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                       sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                       sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.plot(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                       sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax[2].step(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                       sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                       sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                       sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['time']-parValue['start_desorption']/model.get('control_buffer2.scaling'), \
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
      diagrams.append("ax1.plot(sim_res['time'] - parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'] - parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['time'] - parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'] - parValue['start_desorption']/model.get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
      diagrams.append("ax3.step(sim_res['time'] - parValue['start_desorption']/model.get('control_buffer2.scaling'), \
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
      diagrams.append("ax1.plot(sim_res['ackF']/model.get('column.V')[0], \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF']/model.get('column.V')[0], \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
     # diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF']/model.get('column.V')[0], \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF']/model.get('column.V')[0], \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
    # diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
      diagrams.append("ax3.step(sim_res['ackF']/model.get('column.V')[0], \
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
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'], sim_res['column.outlet.c[2]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax3.plot(sim_res['time'], sim_res['column.outlet.c[3]'], label='A', color='r', linestyle=linetype)")

   else:
      print("Plot window type not correct") 

# Define and extend describe for the current application
def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""

   if name == 'chromatography':
      print('Ion exchange chromatorgraphy controlled with varying salt-concentration. The pH is kept constant.')        

   elif name in ['liquidphase', 'media']:
      P = model.get('liquidphase.P')[0]; P_description = model.get_variable_description('liquidphase.P'); 
      P_mw = model.get('liquidphase.mw[1]')[0]
      A = model.get('liquidphase.A')[0]; A_description = model.get_variable_description('liquidphase.A'); 
      A_mw = model.get('liquidphase.mw[2]')[0]
      E = model.get('liquidphase.E')[0]; E_description = model.get_variable_description('liquidphase.E'); 
      E_mw = model.get('liquidphase.mw[3]')[0]
      PS = model.get('liquidphase.PS')[0]; PS_description = model.get_variable_description('liquidphase.PS'); 
      PS_mw = model.get('liquidphase.mw[4]')[0]
      AS = model.get('liquidphase.AS')[0]; AS_description = model.get_variable_description('liquidphase.AS'); 
      AS_mw = model.get('liquidphase.mw[5]')[0]

      print('Chromatography liquidphase (or mobilephase) substances included in the model')
      print()
      print(P_description, '                 - index = ', P, '- molecular weight = ', P_mw, 'Da')
      print(A_description, '      - index = ', A, '- molecular weight = ', A_mw, 'Da')
      print(E_description, '                     - index = ', E, '- molecular weight = ', E_mw, 'Da')
      print(PS_description, '           - index = ', PS, '- molecular weight = ', PS_mw, 'Da')
      print(AS_description, '- index = ', AS, '- molecular weight = ', AS_mw, 'Da')
      print()
      print('Note that both proteins P and A as well as the salt-ion E is modelled to the same mobile phase volume.')

   elif name in ['parts']:
      describe_parts(component_list_minimum)
      
   elif name in ['MSL']:
      describe_MSL()

   else:
      describe_general(name, decimals)
         
#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

FMU_explore_info()