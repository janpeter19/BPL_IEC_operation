# BPL_IEC_operation

This notebook show ion exchange chromatography in laboratory scale.  The model describes separation 
of two proteins, desired product P and an antagonist A. The simulation goes through the differents
steps: equlibration, sample-adsorption, washing 1, desorption, washing 2, and finally equlibration and preparation
for the next batch. Simulation is done using an FMU from Bioprocess Library *for* Modelica. Below a diagramwith a typical 
simulation that you will get at the end of the Jupyter notebook.
 
![](Fig_BPL_IEC_operation.png)

You see in the diagram typical resuts during opoeration.

You start up the notebook in Colab by pressing here
[start BPL notebook](https://colab.research.google.com/github/janpeter19/BPL_IEC_operation/blob/main/BPL_IEC_operation_colab.ipynb)
Then you in the menu choose Runtime/Run all. If you have chosen the altarnative with FMPy click first on the symbol Open in Colab.
The subsequent execution of the simulations of IEC operation take just a second or so. 

You can continue in the notebook and make new simulations and follow the examples given. Here are many things to explore!

Note that:
* The script occassionaly get stuck during installation. Then just close the notebook and start from scratch.
* Runtime warnings are at the moment silenced. The main reason is that we run with an older combination of PyFMI and Python that bring depracation warnings of little interest. 
* Remember, you need to have a google-account!

Just to be clear, no installation is done at your local computer.

Work in progress - stay tuned!

License information:
* The binary-file with extension FMU is shared under the permissive MIT-license
* The other files are shared under the GPL 3.0 license
