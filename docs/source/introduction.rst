Introduction
============

PyProBE is a Python package designed to simplify the process of analysing data from 
battery cyclers. It has four core objectives:

1. Ease of use
    PyProBE is written in Python, a commonly used programming language in the battery
    community. However, it has been designed to be approachable for those with little 
    programming experience. 

    PyProBE breaks down complex battery data into simple, easy to understand objects 
    that can be accessed with a few lines of code using natural language. The 
    procedure shown below:

    .. image:: images/Procedures_and_experiments.png

   can be filtered with simple commands like:

    .. code-block:: python

        # Filter the data to only include the charge pulses
        cell.procedure['RPT'].experiment('Charge Pulses')

        # Filter to only the charge step of the third pulse
        cell.procedure['RPT'].experiment('Charge Pulses').cycle(2).charge(0)

   This makes it easy to quickly access the data you need for analysis.

2. Accelerate battery data exploration
    PyProBE has a built-in :ref:`plotting <plotting>` module for fast and flexible 
    visualisation of battery data. It also includes a graphical user interface (GUI) 
    for exploring data interactively, with almost no code. 

3. Encourage data processing aligned with FAIR principles
    PyProBE is designed to encourage good practice for storing and processing data. The 
    FAIR principles. PyProBE requires a README file to sit alongside your experimental 
    data which is:

    Computer readable
      Simplifies the PyProBE backend, maximises flexibility to different input data and
      makes the setup setup process fast and intuitive for new data.

    Human readable
      Sits alongside your data to allow others to quickly understand your experimental
      procedure.

   See the :ref:`guidance <writing_a_readme_file>` for writing README files for your
   experiments.

4. Host a library of post processing methods
    PyProBE's :mod:`method <~pyprobe.methods>` framework enables further analysis of
    battery data, maintaining compatibility with the PyProBE data format and plotting
    tools while ensuring functions are simply defined, portable and tested.

    It is easy to write your own method. See the :ref:`guidance <creating_a_new_method>`
    and `contributing guidelines <https://github.com/ImperialCollegeLondon/PyProBE/blob/main/CONTRIBUTING.md>`_.