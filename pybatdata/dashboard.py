import streamlit as st
import os
import pandas as pd
import pickle
import sys
import plotly.graph_objects as go

import pandas as pd
# Add the parent directory of pybatdata to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('procedure_dict.pkl', 'rb') as f:
    procedure_dict = pickle.load(f)

metadata = pd.DataFrame(procedure_dict).drop('Data', axis=1)
def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.sidebar.data_editor(
        df_with_selections,
        hide_index=True,  # Keep the index visible
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    selected_indices = selected_rows.index.tolist()  # Get the indices of the selected rows
    return selected_indices

# Display the DataFrame in the sidebar
selected_indices = dataframe_with_selections(metadata)
# Get the names of the selected rows
selected_names = [procedure_dict[i]['Name'] for i in selected_indices]

# Select an experiment
experiment_names = procedure_dict[0]['Data'].titles.keys()
selected_experiment = st.sidebar.selectbox('Select an experiment', experiment_names)

# Get the cycle and step numbers from the user
cycle_step_input = st.sidebar.text_input('Enter the cycle and step numbers (e.g., "cycle(1).step(2)")')
x_options = ['Time (s)', 'Capacity (Ah)']
y_options = ['Voltage (V)', 'Current (A)', 'Capacity (Ah)']

# Create select boxes for the x and y axes
x_axis = st.selectbox('x axis', x_options, index=0)
y_axis = st.selectbox('y axis', y_options, index=1)
secondary_y_axis = st.selectbox('Secondary y axis', ['None'] + y_options, index=0)

# Create a figure
fig = go.Figure()
selected_data = []
for i in range(len(selected_indices)):
    selected_index = selected_indices[i]
    experiment_data = procedure_dict[selected_index]['Data'].experiment(selected_experiment)
    # Check if the input is not empty
    if cycle_step_input:
        # Use eval to evaluate the input as Python code
        filtered_data = eval(f'experiment_data.{cycle_step_input}')
    else: 
        filtered_data = experiment_data
    
    filtered_data = filtered_data.RawData.to_pandas()
    selected_data.append(filtered_data)
    # Add a line to the plot for each selected index
    fig.add_trace(go.Scatter(x=filtered_data[x_axis], 
                             y=filtered_data[y_axis], 
                             mode='lines', 
                             line = dict(color = procedure_dict[selected_index]['Color']),
                             name=procedure_dict[selected_index]['Name'],
                             yaxis='y1'))
    
    # Add a line to the secondary y axis if selected
    if secondary_y_axis != 'None':
        fig.add_trace(go.Scatter(x=filtered_data[x_axis], 
                                 y=filtered_data[secondary_y_axis], 
                                 mode='lines', 
                                 name=procedure_dict[selected_index]['Name'],
                                 yaxis='y2'))

# Set the plot's title and labels
fig.update_layout(xaxis_title=x_axis, 
                  yaxis_title=y_axis,
                  showlegend=True,
                  template='simple_white')

if secondary_y_axis != 'None':
    fig.update_layout(yaxis2=dict(title=secondary_y_axis, overlaying='y', side='right'))

# Show the plot
st.plotly_chart(fig)  

# Show raw data in tabs
if selected_data:
    tabs = st.tabs(selected_names)
    columns = ['Time (s)', 'Cycle', 'Step', 'Current (A)', 'Voltage (V)', 'Capacity (Ah)']
    for tab in tabs:
        tab.dataframe(selected_data[tabs.index(tab)][columns], hide_index=True)
