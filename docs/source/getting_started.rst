Getting Started
===============

Making a cell object
--------------------
PyProBE stores all experimental data and information in a  :class:`pyprobe.cell.Cell` 
object. It has two main attributes: a dictionary of cell details and experimental info 
(:attr:`pyprobe.cell.Cell.info`) and a dictionary of experimental procedures performed 
on the cell (:attr:`pyprobe.cell.Cell.procedure`).

A cell object can be created by providing an info dictionary:

.. code-block:: python

   from pyprobe.cell import Cell # module for storing cell data

   # Describe the cell. Required fields are 'Name'.
   info_dictionary = {'Name': 'Sample cell',
                      'Chemistry': 'NMC622',
                      'Nominal Capacity [Ah]': 0.04,
                      'Cycler number': 1,
                      'Channel number': 1,}

   # Create a cell object
   cell = Cell(info_dictionary)

The info dictionary can contain any number of key-value pairs. The only required key is
'Name', which is used to identify the cell in plots.

Converting data to PyProBE Format
---------------------------------
PyProBE defines a Procedure as a dataset collected from a single run of an experimental
procedure created on a battery cycler. Throughout its life, a cell will likely undergo
multiple procedures, such as beginning-of-life testing, degradation cycles, reference 
performance tests (RPTs) etc. 

Before adding data to a cell object, it must be converted into the PyProBE standard 
format. This is done with the :func:`pyprobe.cell.Cell.process_cycler_file` method:

.. code-block:: python

   # From the previously created cell instance
   cell.process_cycler_file(cycler = 'neware',
                            folder_path = 'path/to/root_folder/experiment_folder',
                            input_filename = 'cycler_file.csv',
                            output_filename = 'processed_cycler_file.parquet')

Your parquet file will be saved in the same directory (:code:`folder_path`) as the input
file. Once converted into this format, PyProBE is agnostic to cycler manufacturer
and model. For more details on the PyProBE standard format, and an up-to-date list of
supported cyclers, see the :ref:`input_data_guidance` section. 

Working with multiple input files
---------------------------------
Some cyclers may output data in multiple files. For example, BioLogic Modulo Bat 
procedures. Assuming the data is all in the same folder, PyProBE is able to collect all
of the files and process them into a single parquet file. This is done by providing a 
:code:`*` wildcard in the :code:`input_filename` argument:

.. code-block:: python

   # From the previously created cell instance
   cell.process_cycler_file(cycler = 'neware',
                            folder_path = 'path/to/root_folder/experiment_folder',
                            input_filename = 'cycler_file*.csv',
                            output_filename = 'processed_cycler_file.parquet')

This will process all files in the folder that match the pattern 
:code:`cycler_file*.csv`, e.g. :code:`cycler_file_1.csv`, :code:`cycler_file_2.csv`, 
etc.

For BioLogic Modulo Bat data:

.. code-block:: python

   cell.process_cycler_file(cycler = 'biologic',
                            folder_path = 'path/to/root_folder/experiment_folder',
                            input_filename = 'cycler_file_*_MB.mpt',
                            output_filename = 'processed_cycler_file.parquet')

Adding data to a cell object
----------------------------
A data file in the standard PyProBE format can be added to a cell object using the
:func:`pyprobe.cell.Cell.add_procedure` method:

.. code-block:: python

   # Add the processed data to the cell object
   cell.add_procedure(procedure_name = 'Example procedure',
                      folder_path = 'path/to/root_folder/experiment_folder',
                      procedure_filename = 'processed_cycler_file.parquet')

Any number of procedures can be added to a cell, for example:

.. code-block:: python

   # Add the first procedure
   cell.add_procedure(procedure_name = 'Cycling',
                      folder_path = 'path/to/root_folder/experiment_folder',
                      procedure_filename = 'processed_cycler_file_cycling.parquet')
   
   # Add the second procedure
   cell.add_procedure(procedure_name = 'RPT',
                      folder_path = 'path/to/root_folder/experiment_folder',
                      procedure_filename = 'processed_cycler_file_RPT.parquet')

   print(cell.procedure)
   # Returns: dict({'Cycling': <pyprobe.procedure.Procedure object…, 'RPT': <pyprobe.procedure.Procedure object…})

Batch preprocessing
-------------------
If you have multiple cells undergoing the same experimental procedures, you can use the
built-in batch processing functionality in PyProBE to speed up your workflow. You must
first create a list of :attr:`pyprobe.cell.Cell` objects.

The fastest way to do this is to store an Experiment Record alongside your data. This is
an Excel file that contains important experimental information about your cells and the
procedures they have undergone. See the :ref:`writing_experiment_record` section for 
guidance.
