# PyProBE
PyProBE is a Python library for the analysis of battery cycling data.

## Using PyProBE
PyProBE should be imported as a package into an external post-processing script or jupyter notebook.

```python
from pybatdata.cell import Cell
```

You can make a PyProBE Cell object with only a dictionary of metadata about the cell being cycled:
```python
info = dict({'metadata item': value})
cell = Cell(info)
```

To add data to your cell object, it must first be converted into the standard format used by PyProBE. See the list of currently supported cyclers and file formats in the documentation.

```puthon
cell.process_cycler_file(cycler = 'supported_cycler',
                         folder_path = 'root_directory/subfolder',
                         file_name = 'cell_data.anyfomat')
```

To add the newly created ```.parquet``` file to the cell object, you must have a ```README.yaml``` file saved alongside your raw data, following the guidance in the documentation.

```python
cell.add_data(title = 'procedure title', 
              input_path = 'root_directory/subfolder', 
              file_name = 'cell_data.parquet')
```

Batch processing can also be done. This requires an ```Experiment_Record.xlsx``` in the ```root directory```, according to the guidelines in the documentation. For more information see the documentation and this example.

## Installing PyProBE
<details>
  <summary>Linux/macOS</summary>

  1. Clone the repository and enter the directory:
  ```bash
  $ git clone
  ```

  2. Create and activate a virtual environment:
  
  venv (in your working directory):
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```
conda (in any directory):
```bash
$ conda create -n pybatdata python=3.12
$ conda activate pybatdata
```


  3. Install PyProBE as a package into your virtual environment:
  ```bash
  $ cd PyProBE
  $ pip install .
  ```
</details>

<details>
  <summary>Windows</summary>

  1. Clone the repository and enter the directory:
  ```bat
  > git clone
  ```

  2. Create and activate a virtual environment:
  
  venv (in your working directory):
```bat
> python -m venv .venv
> source .venv/bin/activate
```
conda (in any directory):
```bash
> conda create -n pybatdata python=3.12
> conda activate pybatdata
```

  3. Install PyProBE as a package into your virtual environment:
  ```bat
  > cd PyProBE
  > pip install .
  ```
</details>

## Citing PyECN

TBC


## Contributing to PyECN

Contributions to PyECN are welcome. Please see the [contributing guidelines](CONTRIBUTING.md).


## License

PyECN is fully open source. For more information about its license, see [LICENSE](LICENSE.md).


## Contributors