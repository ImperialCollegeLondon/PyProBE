import streamlit as st
import os
import pandas as pd
import pickle
import sys
import plotly.graph_objects as go
import plotly

# Add the parent directory of pybatdata to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('dashboard_data.pkl', 'rb') as f:
    cell_list = pickle.load(f)

st.title('PyBatData Dashboard')
st.sidebar.title('Select data to plot')

info_list = []
for i in range(len(cell_list)):
    info_list.append(cell_list[i].info)
info = pd.DataFrame(info_list)
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
selected_indices = dataframe_with_selections(info)
# Get the names of the selected rows
selected_names = [cell_list[i].info['Name'] for i in selected_indices]

# Select a procedure
procedure_names = cell_list[0].procedure.keys()
selected_raw_data = st.sidebar.selectbox('Select a procedure', procedure_names)

# Select an experiment
experiment_names = cell_list[0].procedure[selected_raw_data].titles.keys()
selected_experiment = st.sidebar.selectbox('Select an experiment', experiment_names)

# Get the cycle and step numbers from the user
cycle_step_input = st.sidebar.text_input('Enter the cycle and step numbers (e.g., "cycle(1).step(2)")')
x_options = ['Time (s)', 'Capacity (Ah)', 'Capacity (mAh)', 'Capacity Throughput (Ah)']
y_options = ['Voltage (V)', 'Current (A)', 'Current (mA)', 'Capacity (Ah)', 'Capacity (mAh)']

graph_placeholder = st.empty()

col1, col2, col3 = st.columns(3)
# Create select boxes for the x and y axes
x_axis = col1.selectbox('x axis', x_options, index=0)
y_axis = col2.selectbox('y axis', y_options, index=1)
secondary_y_axis = col3.selectbox('Secondary y axis', ['None'] + y_options, index=0)

# Select plot theme

themes = list(plotly.io.templates)
themes.remove('none')
themes.remove('streamlit')
themes.insert(0, 'default')
plot_theme = 'simple_white'

# Create a figure
fig = go.Figure()
selected_data = []
for i in range(len(selected_indices)):
    selected_index = selected_indices[i]
    experiment_data = cell_list[selected_index].procedure[selected_raw_data].experiment(selected_experiment)
    # Check if the input is not empty
    if cycle_step_input:
        # Use eval to evaluate the input as Python code
        filtered_data = eval(f'experiment_data.{cycle_step_input}')
    else: 
        filtered_data = experiment_data
    
    filtered_data = filtered_data.raw_data.to_pandas()
    selected_data.append(filtered_data)
    # Add a line to the plot for each selected index
    fig.add_trace(go.Scatter(x=filtered_data[x_axis], 
                             y=filtered_data[y_axis], 
                             mode='lines', 
                             line = dict(color = cell_list[selected_index].color),
                             name=cell_list[selected_index].info['Name'],
                             yaxis='y1',
                             showlegend=True))
    
    # Add a line to the secondary y axis if selected
    if secondary_y_axis != 'None':
        fig.add_trace(go.Scatter(x=filtered_data[x_axis], 
                                 y=filtered_data[secondary_y_axis], 
                                 mode='lines', 
                                 line=dict(color=cell_list[selected_index].color, dash='dash'),
                                 name=cell_list[selected_index].info['Name'],
                                 yaxis='y2',
                                 showlegend=False))
if secondary_y_axis != 'None':     
    # Add a dummy trace to the legend to represent the secondary y-axis
    fig.add_trace(go.Scatter(x=[None], 
                            y=[None], 
                            mode='lines', 
                            line=dict(color='black', dash='dash'),
                            name=secondary_y_axis,
                            showlegend=True))

title_font_size = 18
axis_font_size = 14
# Set the plot's title and labels
fig.update_layout(xaxis_title=x_axis, 
                  yaxis_title=y_axis,
                  yaxis2 = dict(title=secondary_y_axis,
                                anchor='free',
                                overlaying='y',
                                autoshift=True,
                                tickmode='sync'),
                  template=plot_theme if plot_theme != 'default' else 'plotly',
                  title_font=dict(size=title_font_size),
                  xaxis_title_font=dict(size=title_font_size),
                  yaxis_title_font=dict(size=title_font_size),
                  xaxis_tickfont=dict(size=axis_font_size),
                  yaxis_tickfont=dict(size=axis_font_size),
                  legend = dict(x=1.2 if secondary_y_axis != 'None' else 1,
                                y = 1,
                                font = dict(size=axis_font_size))
                    )

if secondary_y_axis != 'None':
    fig.update_layout(yaxis2=dict(title=secondary_y_axis, overlaying='y', side='right'),
                      yaxis2_tickfont=dict(size=axis_font_size),
                      yaxis2_title_font=dict(size=title_font_size))

# Show the plot
graph_placeholder.plotly_chart(fig, theme='streamlit' if plot_theme == 'default' else None) 

# Show raw data in tabs
if selected_data:
    tabs = st.tabs(selected_names)
    columns = ['Time (s)', 'Cycle', 'Step', 'Current (A)', 'Voltage (V)', 'Capacity (Ah)']
    for tab in tabs:
        tab.dataframe(selected_data[tabs.index(tab)][columns], hide_index=True)
