#%%

import streamlit as st
import os
import pandas as pd
import pickle
import sys
# Add the parent directory of pybatdata to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('procedure_dict.pkl', 'rb') as f:
    procedure_dict = pickle.load(f)

metadata = pd.DataFrame(procedure_dict).drop('Data', axis=1)
# Display the DataFrame in the sidebar
st.sidebar.write(metadata)

# Create a selectbox for each row in the DataFrame
options = [f'Row {i}' for i in range(len(metadata))]
selected_row = st.selectbox('Select a row', options)

# Convert the selected row label back to an index
selected_index = int(selected_row.split(' ')[1])

experiment_names = procedure_dict[selected_index]['Data'].titles.keys()
selected_experiment = st.selectbox('Select an experiment', experiment_names)
data = procedure_dict[selected_index]['Data'].experiment(selected_experiment)

# Get the cycle and step numbers from the user
cycle_step_input = st.text_input('Enter the cycle and step numbers (e.g., "cycle(1).step(2)")')

# Check if the input is not empty
if cycle_step_input:
    # Use eval to evaluate the input as Python code
    try:
        raw_data = eval(f'data.{cycle_step_input}.RawData')
        st.write(raw_data.to_pandas())
    except Exception as e:
        st.write(f'Error: {e}')
#st.write(procedure_dict[selected_index]['Data'].experiment('pOCV').RawData.to_pandas())


    #%%

