"""Script to create a Streamlit dashboard for PyProBE."""

import copy
import os
import pickle
import platform
import subprocess
from typing import TYPE_CHECKING, Any

import distinctipy
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from pyprobe.cell import Cell

if TYPE_CHECKING:
    from pyprobe.result import Result


def launch_dashboard(cell_list: list[Cell]) -> None:
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


class _Dashboard:
    """Class to create a Streamlit dashboard for PyProBE."""

    def __init__(self, cell_list: list[Cell]) -> None:
        """Initialize the dashboard with the cell list."""
        self.cell_list = cell_list
        self.info = self.get_info(self.cell_list)

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

    @staticmethod
    def get_info(cell_list: list[Cell]) -> pl.DataFrame:
        """Get the cell information from the cell list.

        Args:
            cell_list (list): The list of cell objects.

        Returns:
            The dataframe with the cell information.
        """
        info_list = []
        for i in range(len(cell_list)):
            info_list.append(cell_list[i].info)
        return pl.DataFrame(info_list)

    @staticmethod
    def dataframe_with_selections(df: pl.DataFrame) -> list[int]:
        """Create a dataframe with a selection column for user input.

        Args:
            df (pd.DataFrame): The dataframe to display.

        Returns:
            list: The list of selected row indices.
        """
        df = df.to_pandas()
        df_with_selections = copy.deepcopy(df)
        df_with_selections.insert(0, "Select", False)
        return df_with_selections

    def select_cell_indices(self) -> list[int]:
        """Get dataframe row selections."""
        edited_df = st.sidebar.data_editor(
            self.dataframe_with_selections(self.info),
            hide_index=True,  # Keep the index visible
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=self.info.columns,
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.Select]
        selected_indices = (
            selected_rows.index.tolist()
        )  # Get the indices of the selected rows
        return selected_indices

    def get_common_procedures(self) -> list[str]:
        """Get the common procedure names from the selected cells."""
        procedure_names_sets = [
            list(self.cell_list[i].procedure.keys()) for i in self.selected_indices
        ]

        # Find the common procedure names
        if len(procedure_names_sets) == 0:
            procedure_names: list[str] = []
        else:
            procedure_names = list(procedure_names_sets[0])
            for s in procedure_names_sets[1:]:
                procedure_names = [x for x in procedure_names if x in s]
        return list(procedure_names)

    def select_experiment(self) -> tuple[Any, ...]:
        """Select an experiment from the selected procedure."""
        if self.selected_procedure is not None:
            experiment_names = (
                self.cell_list[self.selected_indices[0]]
                .procedure[self.selected_procedure]
                .experiment_names
            )
            selected_experiment = st.sidebar.multiselect(
                "Select an experiment",
                experiment_names,
            )
            return tuple(selected_experiment)
        else:
            return ()

    def get_data(self) -> list["Result"]:
        """Get the data from the selected cells."""
        selected_data = []
        for i in range(len(self.selected_indices)):
            selected_index = self.selected_indices[i]
            experiment_data: Result
            if len(self.selected_experiments) == 0:
                experiment_data = self.cell_list[selected_index].procedure[
                    self.selected_procedure
                ]
            else:
                experiment_data = (
                    self.cell_list[selected_index]
                    .procedure[self.selected_procedure]
                    .experiment(*self.selected_experiments)
                )
            # Check if the input is not empty
            if self.cycle_step_input:
                # Use eval to evaluate the input as Python code
                filtered_data = eval(f"experiment_data.{self.cycle_step_input}")
            else:
                filtered_data = experiment_data
            selected_data.append(filtered_data)
        return selected_data

    def add_primary_trace(self, data: "Result", color: str) -> None:
        """Add the primary trace to the plot.

        Args:
            data (Result): The data to plot.
            color (str): The color for the trace.
        """
        primary_trace = go.Scatter(
            x=data.get(f"{self.filter_stage} {self.x_axis}".strip()),
            y=data.get(self.y_axis),
            mode="lines",
            name=f"{data.info[self.cell_identifier]}",
            line={"color": color},
        )
        self.fig.add_trace(primary_trace)

    def add_secondary_trace(self, data: "Result", color: str) -> None:
        """Add the secondary trace to the plot.

        Args:
            data (Result): The data to plot.
            color (str): The color for the trace.
        """
        secondary_trace = go.Scatter(
            x=data.get(f"{self.filter_stage} {self.x_axis}".strip()),
            y=data.get(self.secondary_y_axis),
            mode="lines",
            name=f"{data.info[self.cell_identifier]}",
            yaxis="y2",
            line={
                "color": color,
                "dash": "dash",
            },  # Use the same color as the primary trace
            showlegend=False,
        )
        self.fig.add_trace(secondary_trace)

    def add_secondary_y_legend(self) -> None:
        """Add the secondary y-axis legend to the plot."""
        self.fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": "black", "dash": "dash"},
                name=self.secondary_y_axis,
                showlegend=True,
            ),
        )

    def style_fig(self) -> None:
        """Style the plot."""
        title_font_size = 18
        axis_font_size = 14
        default_layout = go.Layout(
            template="simple_white",
            title=None,
            xaxis_title_font={"size": title_font_size},
            yaxis_title_font={"size": title_font_size},
            xaxis_tickfont={"size": axis_font_size},
            yaxis_tickfont={"size": axis_font_size},
            legend_font={"size": axis_font_size},
            legend={"x": 1.2},
            width=800,
            height=600,
        )
        # Update layout for dual-axis
        self.fig.update_layout(
            yaxis={
                "title": self.y_axis,
            },
            yaxis2={"title": self.secondary_y_axis, "overlaying": "y", "side": "right"},
            xaxis={
                "title": self.x_axis,
            },
        )
        self.fig.update_layout(default_layout)

    def run(self) -> None:
        """Run the Streamlit dashboard."""
        st.title("PyProBE Dashboard")
        st.sidebar.title("Select data to plot")
        self.selected_indices = self.select_cell_indices()
        self.selected_procedure = st.sidebar.selectbox(
            "Select a procedure",
            self.get_common_procedures(),
        )
        self.selected_experiments = self.select_experiment()
        self.cycle_step_input = st.sidebar.text_input(
            'Enter the cycle and step numbers (e.g., "cycle(1).step(2)")',
        )
        col1, col2, col3, col4, col5 = st.columns(5)
        self.filter_stage = col1.selectbox(
            "Filter stage",
            ["", "Experiment", "Cycle", "Step"],
            index=0,
        )
        self.x_axis = col2.selectbox("x axis", self.x_options, index=0)
        self.y_axis = col3.selectbox("y axis", self.y_options, index=1)
        self.secondary_y_axis = col4.selectbox(
            "Secondary y axis",
            ["None"] + self.y_options,
            index=0,
        )
        self.cell_identifier = col5.selectbox(
            "Legend label",
            self.info.collect_schema().names(),
        )
        selected_names = [
            self.cell_list[i].info[self.cell_identifier] for i in self.selected_indices
        ]
        selected_data = self.get_data()
        graph_placeholder = st.empty()
        self.fig = go.Figure()
        colors = distinctipy.get_colors(len(self.cell_list), rng=0)
        for i, data in enumerate(selected_data):
            color = distinctipy.get_hex(colors[i])
            self.add_primary_trace(data, color)
            if self.secondary_y_axis != "None":
                self.add_secondary_trace(data, color)

        if self.secondary_y_axis != "None":
            self.add_secondary_y_legend()
        self.style_fig()
        if len(selected_data) > 0 and len(self.selected_procedure) > 0:
            graph_placeholder.plotly_chart(
                self.fig,
                theme="streamlit",  # if plot_theme == "default" else None
            )

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
                tab.dataframe(
                    selected_data[tabs.index(tab)].data.select(columns).to_pandas(),
                    hide_index=True,
                )


if __name__ == "__main__":
    with open("dashboard_data.pkl", "rb") as f:
        cell_list = pickle.load(f)
    _Dashboard(cell_list).run()
