Adding a Cycler
===============

Once parsed into the standard format, PyProBE is agnostic to the source of the data.
However, there is a processing step that is required for data from all cycler to be 
converted into this format. This is done from within the :mod:`pyprobe.cyclers` 
module.

Most of the work is done by the :class:`~pyprobe.cyclers.basecycler.BaseCycler` class.
The cycler-specific modules exist to modify settings, and sometimes override functions
of this class. This guide will go through the steps of making a cycler module to 
allow PyProBE to work with your data.

1. Set up the class. 
   
   The class should inherit from :class:`~pyprobe.cyclers.basecycler.BaseCycler`
   which is a `Pydantic BaseModel <https://docs.pydantic.dev/latest/api/base_model/>`_.
   You should then declare the following variables:
   
   - :code:`input_data_path` as an unset string
         
   - :code:`column_dict` a dictionary relating the cycler format for column headings to
     the PyProBE format. E.g. :code:`{"I/*": Current [*]}`. Note that the units are 
     indicated by an asterisk :code:`*`: PyProBE will perform unit conversion automatically.
   
   Below is an example from the :class:`~pyprobe.cyclers.biologic.Biologic` class:

   .. literalinclude:: ../../../pyprobe/cyclers/biologic.py
    :language: python
    :linenos:
    :lines: 12-25

2. Add a method to read a file into a dataframe. 
   
   The :func:`~pyprobe.cyclers.basecycler.BaseCycler.read_file` method works for the simplest
   example, where the file contents are only the data under column headings. This may work
   for you, or may just require adding to its list of recognized file extensions.

   However, some cyclers, like the Biologic, have large headers which may have data you 
   need to extract. The Biologic class' :func:`~pyprobe.cyclers.biologic.Biologic.read_file`
   method extracts the start time from the header and uses it to form a date column.
   If you need to do the same, simply define your own copy of the `read_file` method in
   the class of your cycler.

3. Add a method to form the complete imported dataframe.

   It is very likely that the base :func:`~pyprobe.cyclers.basecycler.BaseCycler.get_imported_dataframe`
   will work for you, in which case you will not need this function. In the 
   :class:`~pyprobe.cyclers.biologic.BiologicMB` class, this method is used when there
   are multiple `ModuloBat` files being read to concatenate the dataframes while 
   ensuring that the step number does not reduce to zero when the start of one file ends
   and another begins.

4. Add column property overrides as needed. 
   
   The :class:`~pyprobe.cyclers.basecycler.BaseCycler`
   includes properties for reading each column of the data. If any of these need adjusting,
   re-define them in your new cycler class. For example, the :class:`~pyprobe.cyclers.biologic.Biologic` class
   overrides the :attr:`~pyprobe.cyclers.biologic.Biologic.step` property, to add one
   to the step number to change from 0-indexing to 1-indexing.

.. footbibliography::