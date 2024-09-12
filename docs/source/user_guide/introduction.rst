Introduction
============

PyProBE Structure
-----------------
Below is a flowchart of how a PyProBE :class:`~pyprobe.cell.Cell` object is structured
and how the stored data can be filtered:

.. image:: images/Filtering_flowchart.jpg

All data is stored in a :class:`~pyprobe.cell.Cell` object, which contains an
:attr:`~pyprobe.cell.Cell.info` attribute for storing metadata and a 
:attr:`~pyprobe.cell.Cell.procedure` dictionary for storing data for the experimental
procedures run on the cell. These can be further filtered as described in the 
:ref:`filtering` section of the user guide.

Once the data is filtered, it can be processed further with a 
method in the :mod:`~pyprobe.analysis` module or displayed using the built-in 
:mod:`~pyprobe.plot` module. All filters produce objects that are compatible with the 
plotting module, making it easy to visualise the data at any stage of the analysis. Additionally, 
all methods in the analysis module produce a :class:`~pyprobe.result.Result` which can 
be an input to further methods. This is summarised in the flowchart below:

.. image:: images/Result_flowchart.jpg

This documentation
------------------
These docs are generated from the continuous development branch (main) of the PyProBE
repository. If you are using a particular release of PyProBE, you can generate the docs
specific to your release locally by running the following commands:

.. code-block:: bash

    cd PyProBE/docs
    make html

Then navigate to :code:`PyProBE/docs/build/html/` and open :code:`index.html` in your
web browser.

.. footbibliography::