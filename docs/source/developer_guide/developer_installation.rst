Installation
============

To install PyProBE you must be running Python 3.11 or later. It is recommended to use a 
virtual environment to install PyProBE, for example venv or conda.

The steps to install PyProBE with developer settings are as follows:

1. Enter a directory in which you wish to install PyProBE:
   
   .. code-block:: bash

      cd /path/to/your/directory

2. Clone the repository to your local machine. This can be done either from the 
   main PyProBE repository or your own fork. This creates a directory called PyProBE.

   .. code-block:: bash

      git clone https://github.com/ImperialCollegeLondon/PyProBE.git
      cd PyProBE

3. Create and activate a virtual environment.
  
  .. tabs::
      .. tab:: venv

         In your working directory:

         .. code-block:: bash

            python -m venv venv
            source .venv/bin/activate

      .. tab:: conda
            
         In any directory:

         .. code-block:: bash

            conda create -n pyprobe python=3.12
            conda activate pyprobe

3. Install the developer dependencies:
   
   .. code-block:: bash

      cd /path/to/your/directory/PyProBE
      pip install -r requirements-dev.txt

4. Install PyProBE as a package into your virtual environment:
   
   .. code-block:: bash

      pip install -e .

   The :code:`-e` flag installs in "editable" mode, which means changes that you 
   make to the code will be automatically reflected in the package inside your
   virtual environment.

5. Install the pre-commit hooks:

   .. code-block:: bash

      pre-commit install

.. footbibliography::