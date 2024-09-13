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

The ``info`` dictionary can contain any number of key-value pairs. The only required key is
``'Name'``, which is used to identify the cell in plots. PyProBE will verify that the 
``info`` dictionary contains the ``'Name'`` field. If it does not, it will fill this field with
``'Default Name'``.

Converting data to PyProBE Format
---------------------------------
PyProBE defines a Procedure as a dataset collected from a single run of an experimental
protocol created on a battery cycler. Throughout its life, a cell will likely undergo
multiple procedures, such as beginning-of-life testing, degradation cycles, reference 
performance tests (RPTs) etc. 

Before adding data to a cell object, it must be converted into the PyProBE standard 
format. This is done with the :func:`~pyprobe.cell.Cell.process_cycler_file` method:

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

The Biologic Modulo Bat format has its own reader ``'biologic_MB'``:

.. code-block:: python

   cell.process_cycler_file(cycler = 'biologic_MB',
                            folder_path = 'path/to/root_folder/experiment_folder',
                            input_filename = 'cycler_file_*_MB.mpt',
                            output_filename = 'processed_cycler_file.parquet')


.. _adding_data_to_cell:

Adding data to a cell object
----------------------------
For data to be imported into PyProBE, there must be a corresponding :code:`README.yaml`
file in the same directory as the data file. This file contains details of the 
experimental procedure that generated the data. See the :ref:`writing_a_readme_file`
section for guidance.

A data file in the standard PyProBE format can be added to a cell object using the
:func:`~pyprobe.cell.Cell.add_procedure` method. A procedure must be given a name when 
it is imported. Choose something descriptive, so it is easy to distinguish between 
different procedures that have been run on the same cell.

.. code-block:: python

   # Add the processed data to the cell object
   cell.add_procedure(procedure_name = 'Example procedure',
                      folder_path = 'path/to/root_folder/experiment_folder',
                      filename = 'processed_cycler_file.parquet')

Any number of procedures can be added to a cell, for example:

.. code-block:: python

   # Add the first procedure
   cell.add_procedure(procedure_name = 'Cycling',
                      folder_path = 'path/to/root_folder/experiment_folder',
                      filename = 'processed_cycler_file_cycling.parquet')
   
   # Add the second procedure
   cell.add_procedure(procedure_name = 'RPT',
                      folder_path = 'path/to/root_folder/experiment_folder',
                      filename = 'processed_cycler_file_RPT.parquet')

   print(cell.procedure)
   # Returns: dict({'Cycling': <pyprobe.procedure.Procedure object…, 'RPT': <pyprobe.procedure.Procedure object…})

Batch preprocessing
-------------------
If you have multiple cells undergoing the same experimental procedures, you can use the
built-in batch processing functionality in PyProBE to speed up your workflow. You must
first create a list of :attr:`~pyprobe.cell.Cell` objects.

The fastest way to do this is to store an Experiment Record alongside your data. This is
an Excel file that contains important experimental information about your cells and the
procedures they have undergone. See the :ref:`writing_an_experiment_record` section for 
guidance.

Once you have an Experiment Record, you can create a list of cells using the 
:func:`~pyprobe.cell.make_cell_list` function:

.. code-block:: python

   cell_list = pyprobe.make_cell_list(record_filepath = 'path/to/experiment_record.xlsx',
                                      worksheet_name = 'Sample experiment')

This function creates a list of cells, where the :attr:`~pyprobe.cell.Cell.info` 
dictionary is populated with the information from the Experiment Record.

You can then add procedures to each cell in the list. 
:func:`~pyprobe.cell.Cell.add_procedure` includes the functionality to do this 
parametrically. The steps are as follows:

1. Define a function that generates the filename for each cell.
2. Assign the filename generator function to the :code:`filename` argument in 
   :func:`~pyprobe.cell.Cell.add_procedure`.
3. Provide the inputs to the filename generator function in the 
   :code:`filename_inputs` argument. The order of the inputs must match the order of the
   arguments in the filename generator function. These inputs must be keys of the 
   :attr:`~pyprobe.cell.Cell.info` dictionary. This means that they are likely to be 
   column names in the Experiment Record Excel file.

.. code-block:: python

   # Define functions that generates the filename for each cell
   def input_name_generator(cycler, channel):
       return f'cycler_file_{cycler}_{channel}.csv'

   def output_name_generator(cycler, channel):
       return f'processed_cycler_file_{cycler}_{channel}.parquet'

   # Convert the data to PyProBE format and add the procedure to each cell in the list
   for cell in cell_list:
       cell.process_cycler_file(cycler = 'neware',
                                folder_path = 'path/to/root_folder/experiment_folder',
                                input_filename = input_name_generator,
                                output_filename = output_name_generator,
                                filename_inputs = ["Cycler", "Channel"])
                                
       cell.add_procedure(procedure_name = 'Cycling',
                          folder_path = 'path/to/root_folder/experiment_folder',
                          filename = output_name_generator,
                          filename_inputs = ["Cycler", "Channel"])

Adding data not from a cycler
-----------------------------
In your battery experiment, it is likely that you will be collecting data from sources
additional to your battery cycler. This can be added to your :class:`~pyprobe.filters.Procedure`
object after it has been created with its :func:`~pyprobe.filters.Procedure.add_external_data`
method.

The data that you provide must be timeseries, with a column that can be interpreted in
DateTime format. This is usually a string that may appear like: ``"2024-02-29 09:19:58.554"``.
PyProBE will interpolate your data into the time series of the cycling data already there,
so it can be filtered as normal.


.. footbibliography::