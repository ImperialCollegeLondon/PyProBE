# PyBatData
PyBatData is a Python library for the analysis of battery cycling data.

## Using PyBatData
PyBatData should be imported as a package into an external post-processing script or jupyter notebook.

``` 
from pybatdata.cell import Cell
```

You can make a PyBatData Cell object with only a dictionary of metadata about the cell being cycled:
```
info = dict({'metadata item': value})
cell = Cell(info)
```

To add data to your cell object, it must first be converted into the standard format used by PyBatData. See the list of currently supported cyclers in the docs.

```
cell.process_cycler_file(cycler = 'supported_cycler',
                         folder_path = 'root_directory/subfolder',
                         file_name = 'cell_data.any')
```

To add your data to the cell object, you must have a ```README.yaml``` file saved alongside your raw data, following the guidance in the docs.

```
cell.add_data(title = 'procedure title', 
              input_path = 'root_directory/subfolder', 
              file_name = 'cell_data.parquet')
```
