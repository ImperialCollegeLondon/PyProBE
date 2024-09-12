Installation
============

To install PyProBE you must be running Python 3.11 or later. It is recommended to use a 
virtual environment to install PyProBE, for example venv or conda.

You should have two separate directories: 

* one for the PyProBE repository 
* one for your processing script

The steps to install PyProBE are as follows:

1. Enter a directory in which you wish to install PyProBE:
   
   .. code-block:: bash

      cd /path/to/your/directory

2. Clone the repository to your local machine. This creates a directory called PyProBE.

   .. code-block:: bash

      git clone https://github.com/ImperialCollegeLondon/PyProBE.git
      cd PyProBE

   It is recommended to used the tagged releases for a stable version of the code. 
   You can list the available tags with the following command:

   .. code-block:: bash

      git tag

   You can then checkout the desired tag with the following command:
   
   .. code-block:: bash

      git checkout <tag name>

   For example:

   .. code-block:: bash

      git checkout v0.1.0


   To update your installation you can run:
   
   .. code-block:: bash

      git fetch --tags
      git tag
      git checkout <tag name>

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

3. Install PyProBE's dependencies:
   
   .. code-block:: bash

      cd /path/to/your/directory/PyProBE
      pip install -r requirements.txt

4. Install PyProBE as a package into your virtual environment:
   
   .. code-block:: bash

      pip install .

5. In your working directory you can create a new python script or jupyter notebook to 
   process your data. You can import PyProBE into your script as follows:

   .. code-block:: python

      import pyprobe

.. footbibliography::