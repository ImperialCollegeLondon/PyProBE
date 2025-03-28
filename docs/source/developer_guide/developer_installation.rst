Installation
============

The steps to install PyProBE with developer settings are as follows:

**Clone the repository**

1. Enter a directory in which you wish to install PyProBE:
   
   .. code-block:: bash

      cd /path/to/your/directory

2. Clone the repository to your local machine. This can be done either from the 
   main PyProBE repository or your own fork. This creates a directory called PyProBE.

   .. code-block:: bash

      git clone https://github.com/ImperialCollegeLondon/PyProBE.git
      cd PyProBE

**Installation with uv (recommended)**

To guarantee a safe installation with compatible packages, it is recommended to use
the `uv <https://docs.astral.sh/uv/>`_ tool.

First, follow the steps in the 
`uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_ 
to install uv onto your system. Note: while uv can be installed into a virtual 
environment with pip, it is recommended to install system-wide.

Once uv is installed we can continue with the PyProBE installation. With a single
command, uv installs PyProBE alongside Python and all of its dependencies in a 
virtual environment:

.. code-block:: bash

      uv sync --all-extras

The virtual environment is stored in the :code:`PyProBE/.venv` directory inside your and
can be activated with :code:`source .venv/bin/activate`.

To run a jupyter notebook in VSCode from this environment you can run:

.. code-block:: bash

      uv run -m ipykernel install --user --name <your-chosen-name>

This will create a kernel that you can select within VSCode in the usual way. To use
:code:`%pip` within a jupyter notebook, run :code:`uv venv --seed` prior to the two 
commands above.

.. dropdown:: Alternative installation with pip (not recommended)

   Alternatively, it is possible to install PyProBE with pip. This method skips the
   dependency locking mechanism within uv, so reliable results are not guaranteed if
   this method results in dependency conflicts:

   1. Create and activate a virtual environment.

   2. Install PyProBE as a package into your virtual environment with the developer 
      dependencies:
      
      .. code-block:: bash

         pip install -e '.[dev, docs]'

      The :code:`-e` flag installs in "editable" mode, which means changes that you 
      make to the code will be automatically reflected in the package inside your
      virtual environment.

**Install the pre-commit hooks:**

.. code-block:: bash

   pre-commit install

.. footbibliography::