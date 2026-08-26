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
protocol created on a battery cycler. Throughout its life, a cell undergoes multiple
procedures, such as beginning-of-life testing, degradation cycles, and reference
performance tests (RPTs).

:func:`~pyprobe.io.process_cycler` reads a raw cycler file, normalises it to the
PyProBE column format, and writes it to a ``.parquet`` file next to the source:

.. code-block:: python

   from pyprobe.io import process_cycler

   output_path = process_cycler("path/to/cycler_file.csv")

By default, the written file sits beside the source with the ``.bdf.parquet`` suffix.
Pass a path to the :code:`output_path` argument to control this. The first call takes
longer than a later one, because it runs the conversion. A later call with the same
:code:`output_path` returns the cached path immediately, unless :code:`overwrite_data=True`
forces a fresh conversion.

:meth:`~pyprobe.cell.Cell.add_procedure` then loads the written file into a cell, under
the name you give it:

.. code-block:: python

   import pyprobe

   cell = pyprobe.Cell()
   cell.add_procedure("Sample", output_path)

:meth:`~pyprobe.cell.Cell.add_procedure` also accepts a raw cycler file, a PyProBE
``.parquet`` artifact, a :class:`~pyprobe.filters.Procedure`, a
:class:`~polars.LazyFrame`, or a :class:`~polars.DataFrame` directly. It routes the
source through :meth:`~pyprobe.filters.Procedure.load`, so the two calls above collapse
to one:

.. code-block:: python

   cell.add_procedure("Sample", "path/to/cycler_file.csv")

Any number of procedures can be added to a cell, for example:

.. code-block:: python

   cell.add_procedure("Cycling", "path/to/cycler_file_cycling.csv")
   cell.add_procedure("RPT", "path/to/cycler_file_RPT.csv")

   print(cell.procedure)
   # Returns: {'Cycling': <pyprobe.filters.Procedure object…, 'RPT': <pyprobe.filters.Procedure object…}

Loading a procedure directly
----------------------------
:meth:`~pyprobe.filters.Procedure.load` is the one read path that
:func:`~pyprobe.io.process_cycler` and :meth:`~pyprobe.cell.Cell.add_procedure` both
use underneath. It routes on the type of its *source*:

- A :class:`~polars.LazyFrame`, a :class:`~polars.DataFrame`, or a pandas
  ``DataFrame`` loads directly, with an empty metadata record. A ``column_map``
  names the source column of each BDF column, keyed by the BDF output name in its
  ``"Quantity / unit"`` form, for example ``{"Current / A": "I_meas"}``.
- A ``.parquet`` path reads the file that :func:`~pyprobe.io.process_cycler` or
  :meth:`~pyprobe.result.Table.save` wrote, together with its
  ``<stem>.metadata.json`` sidecar.
- Any other path reads through the cycler plugin that ``bdf`` detects, or the plugin
  that the ``plugin`` argument names.

.. code-block:: python

   from pyprobe.filters import Procedure

   procedure = Procedure.load("path/to/cycler_file.bdf.parquet")

``extra_columns`` names a source column that the BDF ontology does not resolve, on a
``.parquet`` path or a raw file path. Its key is the source column name, and its value
is the name the loaded column carries:

.. code-block:: python

   procedure = Procedure.load(
      "path/to/cycler_file.csv",
      extra_columns={"Pressure(kPa)": "Ambient Pressure / kPa"},
   )

A name given through ``extra_columns`` or ``column_map`` must satisfy the column name
rule described below. A save then writes that name into the artifact, so a later load
of that artifact keeps the column without repeating the map.

Which columns a load keeps
--------------------------
A load reads with every column the source holds, and then reduces the frame. It keeps
every core BDF column that :meth:`~pyprobe.filters.Procedure.load` guarantees or
derives, such as ``Current / A``, ``Voltage / V``, and a time column. It also keeps
every other column whose name carries a ``"Quantity / unit"`` form that
:func:`~pyprobe.columns.is_valid_column_name` accepts.

A load drops every column whose name fails the rule, and logs one warning that names
each dropped column. A user therefore names an extra column once, through
``extra_columns`` or ``column_map``, in a form the rule accepts, and every later load
of the written artifact keeps that column without repeating the map.

When a procedure is loaded, PyProBE performs no README discovery of its own. A legacy
README describing the experimental protocol attaches through
:meth:`~pyprobe.filters.Procedure.attach_legacy_readme`. See the
:ref:`writing_a_readme_file` section for guidance on writing the README, and the
:ref:`the_test_protocol` section for guidance on how PyProBE represents the attached
protocol.

Without an attached protocol, the data is still filterable by cycle, step, and step
type, but not by experiment.

Working with multiple input files
---------------------------------
Some cyclers output data in multiple files, for example BioLogic Modulo Bat
procedures. Assuming the data is all in the same folder, PyProBE collects the files
and processes them into a single ``.parquet`` file. Provide a :code:`*` wildcard in
the source path:

.. code-block:: python

   output_path = process_cycler("path/to/cycler_file*.csv")

This processes every file in the folder that matches the pattern
:code:`cycler_file*.csv`, for example :code:`cycler_file_1.csv`,
:code:`cycler_file_2.csv`, and so on. :func:`~pyprobe.io.process_cycler` loads each
file, extends the first with the rest, and saves the result. See
`Extending a procedure`_ below for the rules that the extend applies.

Saving a procedure
------------------
:meth:`~pyprobe.result.Table.save` writes any :class:`~pyprobe.result.Table`, so a
filtered or a joined :class:`~pyprobe.filters.Procedure` writes the same way a freshly
loaded one does:

.. code-block:: python

   procedure = Procedure.load("path/to/cycler_file.csv")
   procedure.save("path/to/output.bdf.parquet")

The call writes the Parquet data file and the ``<stem>.metadata.json`` sidecar
together. It raises :class:`FileExistsError` where the data file already exists,
unless :code:`overwrite=True` is given, and it raises :class:`ValueError` where the
path does not end with ``.parquet``.

Extending a procedure
---------------------
:meth:`~pyprobe.rawdata.CyclingData.extend` combines two cycling data objects
vertically, and every filtered or loaded :class:`~pyprobe.filters.Procedure` inherits
it. It orders the sources by their first ``Unix Time / s`` value by default, then
applies two rules across the boundary between one source and the next:

- The ``time`` argument controls the ``Test Time / s`` column. ``"continue"``
  (default) adds the last value of one source to the next, so the test time runs on
  without a gap. ``"elapsed"`` derives the test time from ``Unix Time / s`` instead,
  so a real gap between two files survives. ``"keep"`` stacks the recorded values
  verbatim.
- The ``step_id`` argument controls the ``Step ID`` column. ``"offset"`` (default)
  adds the maximum value of one source to the next, so a step identifier stays
  unique across the extended frame. ``"keep"`` stacks the recorded values verbatim.

``Step Count / 1`` is always rebuilt over the whole extended frame, because a recorded
step count resets at a file boundary:

.. code-block:: python

   first = Procedure.load("path/to/cycler_file_1.csv")
   second = Procedure.load("path/to/cycler_file_2.csv")
   first.extend(second)

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
object after it has been created with its :func:`~pyprobe.result.Table.add_data`
method.

The data that you provide must be timeseries, with a column that can be interpreted in
DateTime format. This is usually a string that may appear like: ``"2024-02-29 09:19:58.554"``.
PyProBE will interpolate your data into the time series of the cycling data already there,
so it can be filtered as normal.

Each added column name must satisfy the column name rule described above. A name that
fails the rule raises a :class:`ValueError` that names the column, rather than being
silently dropped later.


.. footbibliography::
