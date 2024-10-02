"""Script to create a Streamlit dashboard for PyProBE."""
import copy
import os
import pickle
import platform
import subprocess
from typing import List

import plotly
import polars as pl
import streamlit as st
from ordered_set import OrderedSet

from pyprobe.cell import Cell
from pyprobe.plot import Plot


def launch_dashboard(cell_list: List[Cell]) -> None:
    """Function to launch the dashboard for the preprocessed data.

    Args:
        cell_list (list): The list of cell objects to display in the dashboard.
    """
    with open("dashboard_data.pkl", "wb") as f:
        pickle.dump(cell_list, f)

    if platform.system() == "Windows":
        subprocess.Popen(
            [
                "cmd",
                "/c",
                "start",
                "/B",
                "streamlit",
                "run",
                os.path.join(os.path.dirname(__file__), "dashboard.py"),
                ">",
                "nul",
                "2>&1",
            ],
            shell=True,
        )
    elif platform.system() == "Darwin":
        subprocess.Popen(
            [
                "nohup",
                "streamlit",
                "run",
                os.path.join(os.path.dirname(__file__), "dashboard.py"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    with open("dashboard_data.pkl", "rb") as f:
        cell_list = pickle.load(f)

    st.title("PyProBE Dashboard")
    st.sidebar.title("Select data to plot")

    info_list = []
    for i in range(len(cell_list)):
        info_list.append(cell_list[i].info)
    info = pl.DataFrame(info_list)

    def dataframe_with_selections(df: pl.DataFrame) -> List[int]:
        """Create a dataframe with a selection column for user input.

        Args:
            df (pd.DataFrame): The dataframe to display.

        Returns:
            list: The list of selected row indices.
        """
        df = df.to_pandas()
        df_with_selections = copy.deepcopy(df)
        df_with_selections.insert(0, "Select", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = st.sidebar.data_editor(
            df_with_selections,
            hide_index=True,  # Keep the index visible
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns.tolist(),
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.Select]
        selected_indices = (
            selected_rows.index.tolist()
        )  # Get the indices of the selected rows
        return selected_indices

    # Display the DataFrame in the sidebar
    selected_indices = dataframe_with_selections(info)
    # Get the names of the selected rows
    selected_names = [cell_list[i].info["Name"] for i in selected_indices]
    # Get the procedure names from the selected cells
    procedure_names_sets = [
        OrderedSet(cell_list[i].procedure.keys()) for i in selected_indices
    ]

    # Find the common procedure names
    if len(procedure_names_sets) == 0:
        procedure_names: List[str] = []
    else:
        procedure_names = list(procedure_names_sets[0])
        for s in procedure_names_sets[1:]:
            procedure_names = [x for x in procedure_names if x in s]
    procedure_names = list(procedure_names)
    selected_raw_data = st.sidebar.selectbox("Select a procedure", procedure_names)

    # Select an experiment
    if selected_raw_data is not None:
        experiment_names = (
            cell_list[selected_indices[0]].procedure[selected_raw_data].experiment_names
        )
        selected_experiment = st.sidebar.multiselect(
            "Select an experiment", experiment_names
        )
        selected_experiment_tuple = tuple(selected_experiment)

    # Get the cycle and step numbers from the user
    cycle_step_input = st.sidebar.text_input(
        'Enter the cycle and step numbers (e.g., "cycle(1).step(2)")'
    )
    x_options = [
        "Time [s]",
        "Time [min]",
        "Time [hr]",
        "Capacity [Ah]",
        "Capacity [mAh]",
        "Capacity Throughput [Ah]",
    ]
    y_options = [
        "Voltage [V]",
        "Current [A]",
        "Current [mA]",
        "Capacity [Ah]",
        "Capacity [mAh]",
    ]

    graph_placeholder = st.empty()

    col1, col2, col3, col4 = st.columns(4)
    # Create select boxes for the x and y axes
    filter_stage = col1.selectbox(
        "Filter stage", ["", "Experiment", "Cycle", "Step"], index=0
    )
    x_axis = col2.selectbox("x axis", x_options, index=0)
    x_axis = f"{filter_stage} {x_axis}".strip()
    y_axis = col3.selectbox("y axis", y_options, index=1)
    secondary_y_axis = col4.selectbox("Secondary y axis", ["None"] + y_options, index=0)

    # Select plot theme

    themes = list(plotly.io.templates)
    themes.remove("none")
    themes.remove("streamlit")
    themes.insert(0, "default")
    plot_theme = "simple_white"

    # Create a figure
    fig = Plot()

    selected_data = []
    for i in range(len(selected_indices)):
        selected_index = selected_indices[i]
        if len(selected_experiment_tuple) == 0:
            experiment_data = cell_list[selected_index].procedure[selected_raw_data]
        else:
            experiment_data = (
                cell_list[selected_index]
                .procedure[selected_raw_data]
                .experiment(*selected_experiment_tuple)
            )
        # Check if the input is not empty
        if cycle_step_input:
            # Use eval to evaluate the input as Python code
            filtered_data = eval(f"experiment_data.{cycle_step_input}")
        else:
            filtered_data = experiment_data

        if secondary_y_axis == "None":
            secondary_y_axis = None

        fig = fig.add_line(filtered_data, x_axis, y_axis, secondary_y=secondary_y_axis)
        filtered_data = filtered_data.data.to_pandas()
        selected_data.append(filtered_data)

    # Show the plot
    if len(selected_data) > 0 and len(procedure_names) > 0:
        graph_placeholder.plotly_chart(
            fig.fig, theme="streamlit" if plot_theme == "default" else None
        )

    # Show raw data in tabs
    if selected_data:
        tabs = st.tabs(selected_names)
        columns = [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
        for tab in tabs:
            tab.dataframe(selected_data[tabs.index(tab)][columns], hide_index=True)
