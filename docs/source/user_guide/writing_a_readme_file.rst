.. _writing_a_readme_file:

Writing a README file
=====================
README files are important to store alongside your experimental data. The PyProBE
README format is a :code:`.yaml` file that contains details about the instructions 
provided to the battery cycler used to generate the data. It is then used filter the 
procedure into experiments that can be analysed separately. 

The :code:`README.yaml` contains the following information:

* The name of the experiment
   This enables filtering by :code:`'Experiment Name'` with the 
   
   :code:`cell.procedure['Procedure Name'].experiment['Experiment Name']` 
   
   syntax.
* Type of experiment
   This enables the creation of different objects for different types of experiments.
   Different experiments enable shortcuts to different analysis. For example, the 
   :class:`~pyprobe.experiments.pulsing.Pulsing` experiment, has a 
   :attr:`~pyprobe.experiments.pulsing.Pulsing.summary` method that summarises the
   resistance of the cell at each pulse.
   The currently supported experiment types are:

   * General
   * Constant Current
   * SOC Reset
   * Cycling
   * Pulsing
   * pOCV

* Steps
   This is a list of strings that describe the each step of the experiment. The strings
   should follow `PyBaMM's Experiment string <https://docs.pybamm.org/en/stable/source/api/experiment/experiment_steps.html#pybamm.step.string>`_ 
   syntax.
* Repeat
   This is the number of times the experiment is repeated. This is useful for cycling
   experiments, where the same steps are repeated multiple times.
   Repeat information should be included in the Steps list if the cycler considers
   the repeat instruction as a seperate step. For example, in the Neware procedure shown
   below, the cycle instruction is step 8.
   
The following is an example of a :code:`README.yaml` file:

.. code-block:: yaml

   # This is the name of the experiment
   Initial Charge:
      # The type of the experiment
      Type: SOC Reset
      # The step numbers in the experiment
      Steps: 
         # Description of each step, using PyBaMM Experiment syntax
         - Rest for 4 hours
         - Charge at constant current of 4mA and constant voltage of 4.2V until current drops to 0.04A
         - Rest for 2 hours
   Break-in Cycles:
      Type: Cycling
      Steps: 
         - Discharge at constant current of 4mA until voltage reaches 3V
         - Rest for 2 hours
         - Charge at constant current of 4mA and constant voltage of 4.2V until current drops to 0.04A
         - Rest for 2 hours
         - Repeat: 5 # the number of times the experiment is repeated
   Discharge Pulses:
      Type: Pulsing
      Steps: 
         - Rest for 10 seconds
         - Discharge at constant current of 20mA until 4mAh has passed or voltage reaches 3V
         - Rest for 30 minutes
         - Rest for 1.5 hours
         - Repeat: 10

Which corresponds to the following Neware procedure file:

.. image:: images/Neware_procedure.png

Shortcuts
---------
It is in the future plans of PyProBE to include automatic generation of the README file
from cycler procedure files. However, for now it must be done manually. For most 
experiments this should not be too time consuming, and provides value by documenting
your data in a human readable format.

For testing purposes or if an experiment is particularly cumbersome to write out. You 
can instead the :code:`Steps` list with :code:`Total Steps`, allowing you to provide 
just a single number for the total number of steps in the experiment:

.. code-block:: yaml

   # This is the name of the experiment
   Initial Charge:
      # The type of the experiment
      Type: SOC Reset
      # The total number of steps in the experiment
      Total Steps: 3

.. footbibliography::