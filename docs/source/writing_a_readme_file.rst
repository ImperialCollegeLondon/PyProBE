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
   See :mod:`pyprobe.experiments` for the currently available experiment types and their
   custom attributes.
* Step numbers 
   This is an optional field, but is required when the step numbers of the data are not
   sequential. In the example below, for instance, step 8 is missing.
* Steps
   This is a list of strings that describe the each step of the experiment. The strings
   should follow `PyBaMM's Experiment string <https://docs.pybamm.org/en/stable/source/api/experiment/experiment_steps.html#pybamm.step.string>`_ 
   syntax.
* Repeat
   This is the number of times the experiment is repeated. This is useful for cycling
   experiments, where the same steps are repeated multiple times.
   
The following is an example of a :code:`README.yaml` file:

.. code-block:: yaml

   # This is the name of the experiment
   Initial Charge:
      # The type of the experiment
      Type: SOC Reset
      # The step numbers in the experiment
      Step Numbers: [1, 2, 3]
      Steps: 
         # Description of each step, using PyBaMM Experiment syntax
         - Rest for 4 hours
         - Charge at constant current of 4mA and constant voltage of 4.2V until current drops to 0.04A
         - Rest for 2 hours
      # The number of times the experiment is repeated
      Repeat: 1
   Break-in Cycles:
      Type: Cycling
      Step Numbers: [4, 5, 6, 7]
      Steps: 
         - Discharge at constant current of 4mA until voltage reaches 3V
         - Rest for 2 hours
         - Charge at constant current of 4mA and constant voltage of 4.2V until current drops to 0.04A
         - Rest for 2 hours
      Repeat: 5
   Discharge Pulses:
      Type: Pulsing
      Step Numbers: [9, 10, 11, 12]
      Steps: 
         - Rest for 10 seconds
         - Discharge at constant current of 20mA until 4mAh has passed or voltage reaches 3V
         - Rest for 30 minutes
         - Rest for 1.5 hours
      Repeat: 10