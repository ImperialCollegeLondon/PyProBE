Importing Data
==============

Making a cell object
--------------------
PyProBE stores all experimental data and information in a :class:`pyprobe.cell.Cell` 
object. It has two main attributes: 

- a dictionary of cell details and experimental info (:attr:`pyprobe.cell.Cell.info`) 
- a dictionary of experimental procedures performed on the cell (:attr:`pyprobe.cell.Cell.procedure`).

A cell object can be created by providing an info dictionary as a keyword argument to 
``info``:

.. code-block:: python

   import pyprobe

   # Describe the cell. Required fields are 'Name'.
   info_dictionary = {'Name': 'Sample cell',
                      'Chemistry': 'NMC622',
                      'Nominal Capacity [Ah]': 0.04,
                      'Cycler number': 1,
                      'Channel number': 1,}

   # Create a cell object
   cell = pyprobe.Cell(info = info_dictionary)

The ``info`` dictionary can contain any number of key-value pairs that provide 
metadata to identify the cell and the conditions it was tested under.

.. _adding_data_to_cell:

Importing data from a cycler
----------------------------
PyProBE defines a Procedure as a dataset collected from a single run of an experimental
protocol created on a battery cycler. Throughout its life, a cell will likely undergo
multiple procedures, such as beginning-of-life testing, degradation cycles, reference 
performance tests (RPTs) etc. 

Data can be imported into PyProBE from a range of cyclers. You import data from a cycler
using the :func:`~pyprobe.cell.Cell.import_from_cycler` method:

.. code-block:: python

   # From the previously created cell instance
   cell.import_from_cycler(
      procedure_name="Sample",
      cycler="neware",
      input_data_path="path/to/cycler_file.csv")

This will do two things:
1. Convert the data in the cycler file into the PyProBE standard format, saved to a 
'.parquet' file. By default, the file will be saved to the same path with only the
file extension changed, however this can be controlled by passing a filepath to the
:code:`output_data_path` argument of this function.
2. Import the data into a the cell's :code:`procedure` dictionary with the given name as
the dictionary key.

These steps can be separated. You can perform step 1 with the :func:`pyprobe.cell.process_cycler_data`
method, and step 2 with the :func:`pyprobe.cell.Cell.import_data` method.

The first time this method is called will take longer than subsequent calls as the data
conversion is executed. Once the '.parquet' file is written future executions will be
much faster.

Any number of procedures can be added to a cell, for example:

.. code-block:: python

   # Add the first procedure
   cell.import_from_cycler(
      procedure_name="Cycling",
      cycler="neware",
      input_data_path="path/to/cycler_file_cycling.csv")
   
   # Add the second procedure
   cell.import_from_cycler(
      procedure_name="RPT",
      cycler="neware",
      input_data_path="path/to/cycler_file_RPT.csv")

   print(cell.procedure)
   # Returns: dict({'Cycling': <pyprobe.procedure.Procedure object…, 'RPT': <pyprobe.procedure.Procedure object…})

When the data is imported, PyProBE will look for a README file in the directory of the
cycler file and/or the PyProBE format '.parquet' file. You can also specify a custom
path for it in the :code:`readme_path` argument. The README file contains details of the 
experimental procedure that generated the data. See the :ref:`writing_a_readme_file`
section for guidance.

Without a README file, the data will still be imported, but will not be filterable
by 'experiment' or by complex cycle patterns.

Working with multiple input files
---------------------------------
Some cyclers may output data in multiple files. For example, BioLogic Modulo Bat 
procedures. Assuming the data is all in the same folder, PyProBE is able to collect all
of the files and process them into a single parquet file. This is done by providing a 
:code:`*` wildcard in the :code:`input_filename` argument:

.. code-block:: python

   # From the previously created cell instance
   cell.import_from_cycler(
      procedure_name="Sample",
      cycler="neware",
      input_data_path="path/to/cycler_file*.csv")

This will process all files in the folder that match the pattern 
:code:`cycler_file*.csv`, e.g. :code:`cycler_file_1.csv`, :code:`cycler_file_2.csv`, 
etc.

The Biologic Modulo Bat format has its own reader ``'biologic_MB'``:

.. code-block:: python

   cell.import_from_cycler(
      procedure_name="biologic_MB",
      cycler="neware",
      input_data_path="path/to/cycler_file_*_MB.mpt")


Batch preprocessing
-------------------
If you have multiple cells undergoing the same experimental procedures, you can create 
a list of :attr:`~pyprobe.cell.Cell` objects together with the 
:func:`~pyprobe.cell.make_cell_list` function.

This requires an Experiment Record alongside your data. This is
an Excel file that contains important experimental information about your cells and the
procedures they have undergone. See the :ref:`writing_an_experiment_record` section for 
guidance.

.. code-block:: python

   cell_list = pyprobe.make_cell_list(record_filepath = 'path/to/experiment_record.xlsx',
                                      worksheet_name = 'Sample experiment')

This function creates a list of cells, where the :attr:`~pyprobe.cell.Cell.info` 
dictionary is populated with the information from the Experiment Record. You can then
loop through these cells, adding data to procedures. It is often helpful to include
parameters of your data file names in the experiment record, so that these can be generated
automatically within your loop.

Adding data not from a cycler
-----------------------------
In your battery experiment, it is likely that you will be collecting data from sources
additional to your battery cycler. This can be added to your :class:`~pyprobe.filters.Procedure`
object after it has been created with its :func:`~pyprobe.result.Result.add_data`
method.

The data that you provide must be timeseries, with a column that can be interpreted in
DateTime format. This is usually a string that may appear like: ``"2024-02-29 09:19:58.554"``.
PyProBE will interpolate your data into the time series of the cycling data already there,
so it can be filtered as normal.


.. footbibliography::