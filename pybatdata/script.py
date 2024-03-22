#%%
import import_data
import postprocessor
import polars as pl
import os

folder = r'/mnt/c/Users/tjh17/OneDrive - Imperial College London/PhD Workspace/05 - Parallel SLP/DATA'
test_name = 'SLP_Cell-Cell_Variation_R1'
filepath = os.path.join(folder, f"{test_name}/SLP_Cell-Cell_Variation_R1_4_1.csv")
#df = pl.scan_csv(filepath, ignore_errors=True)
#print(df.columns)
data = postprocessor.DataLoader.from_parquet(folder, test_name, "Cell_1")
print(data[0].data.collect())
# %%
