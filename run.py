#%%
import os
import sys
sys.path.insert(0, '/home/tjh17/GitRepo/')
import pandas as pd
import pickle
import time
from PyBatDATA.import_data import Neware

def process_csv_files(folder_path):
    # Use glob to match the pattern '*.csv', meaning any file ending with .csv
    for csv_file in os.path.join(folder_path, '*.csv'):
        print(f"Processing {csv_file}")
        # Here you can add code to process the csv file

def open_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = {}  # Create a new data object if the file doesn't exist
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    return data    

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f) 
        
def update_dict_with_pickle(cell_dict, test_name, file_path):
    with open(file_path, 'rb') as f:
        new_data = pickle.load(f)
    cell_dict[test_name] = new_data  # Overwrite or append new_data to cell_dict
    return cell_dict

#folder_path = input("Please enter the folder path: ")
folder_path = "/mnt/c/Users/tjh17/OneDrive - Imperial College London/PhD Workspace/05 - Parallel SLP/DATA"
#experiment = input("Please enter the experiment name:")
test_name = "SLP_Cell-Cell_Variation_R1"
record_xlsx = os.path.join(folder_path, "Experiment_Record.xlsx")
record = pd.read_excel(record_xlsx, sheet_name = test_name)


for i in range(len(record)):
    
    test_details = record.iloc[i].to_dict()
    round = int(test_details['Round'])
    cell_num = int(test_details['Cell'])
    cycler = int(test_details['Cycler']) if not pd.isna(test_details['Cycler']) else None
    channel = int(test_details['Channel']) if not pd.isna(test_details['Channel']) else None
    if cycler == None or channel == None:
        print(f"Skipping: Cell {cell_num}", end=' ')
        continue
    pickle_path = os.path.join(folder_path, f"Cell_{cell_num}.pkl")
    cell_dict = open_pickle(pickle_path)
    t1 = time.time()
    test_data = {}
    test_data["Data"] = Neware.load_data(os.path.join(folder_path, test_name, f"{test_name}_{cycler}_{channel}.csv"))
    cell_dict[test_name] = {**test_details, **test_data}
    save_pickle(cell_dict, pickle_path)
    print(f"Processed: Cell {cell_num} in {time.time()-t1:.2f} seconds. File size = {os.path.getsize(pickle_path)/1e6:.2f} MB") 
    
    