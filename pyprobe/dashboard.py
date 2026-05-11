"""Script to create a Streamlit dashboard for PyProBE."""

import copy
import os
import pickle
import platform
import subprocess
from typing import TYPE_CHECKING, Any

import distinctipy
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

if TYPE_CHECKING:
    import pandas as pd

from pyprobe.cell import Cell
from pyprobe.columns import BDF
from pyprobe.rawdata import RawData


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

    _x_quantity_options: list[str] = ["Test Time", "Net Capacity"]
    _y_quantity_options: list[str] = ["Voltage", "Current", "Net Capacity"]
    _quantity_unit_options: dict[str, list[str]] = {
        "Test Time": ["s", "min", "hr"],
        "Voltage": ["V", "mV"],
        "Current": ["A", "mA"],
        "Net Capacity": ["Ah", "mAh"],
    }
    _display_columns_all = [
        "Test Time / s",
        "Step Index / 1",
        "Current / A",
        "Voltage / V",
        "Net Capacity / Ah",
    ]

    @property
    def x_quantity_options(self) -> list[str]:
        """Get available x-axis quantity options based on selected data."""
        if not hasattr(self, "selected_data") or not self.selected_data:
            return self._x_quantity_options
        canonical = {
            q: f"{q} / {self._quantity_unit_options[q][0]}"
            for q in self._x_quantity_options
        }
        return [
            q
            for q in self._x_quantity_options
            if self.selected_data[0].columns.can_resolve(canonical[q])
        ]

    @property
    def y_quantity_options(self) -> list[str]:
        """Get available y-axis quantity options based on selected data."""
        if not hasattr(self, "selected_data") or not self.selected_data:
            return self._y_quantity_options
        canonical = {
            q: f"{q} / {self._quantity_unit_options[q][0]}"
            for q in self._y_quantity_options
        }
        return [
            q
            for q in self._y_quantity_options
            if self.selected_data[0].columns.can_resolve(canonical[q])
        ]

    @property
    def x_axis(self) -> str:
        """Full column string for the x axis."""
        return f"{self.x_quantity} / {self.x_unit}"

    @property
    def y_axis(self) -> str:
        """Full column string for the primary y axis."""
        return f"{self.y_quantity} / {self.y_unit}"

    @property
    def secondary_y_axis(self) -> str:
        """Full column string for the secondary y axis, or 'None'."""
        if self.secondary_y_quantity == "None":
            return "None"
        return f"{self.secondary_y_quantity} / {self.secondary_y_unit}"

    @staticmethod
    def _resolve_available_columns(
        data: RawData, column_options: list[str]
    ) -> list[str]:
        """Filter column options to only those that can be resolved from data.

        Args:
            data: A Result object with columns metadata.
            column_options: List of potential column names to filter.

        Returns:
            List of column names that can be resolved from the data.
        """
        return [col for col in column_options if data.columns.can_resolve(col)]

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
            info_list.append(getattr(cell_list[i], "info", {}))
        return pl.DataFrame(info_list)

    @staticmethod
    def dataframe_with_selections(df: pl.DataFrame) -> "pd.DataFrame":
        """Create a dataframe with a selection column for user input.

        Args:
            df: The dataframe to display.

        Returns:
            The dataframe with a prepended 'Select' column.
        """
        df_pandas = df.to_pandas()
        df_with_selections = copy.deepcopy(df_pandas)
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

    def get_data(self) -> list[RawData]:
        """Get the data from the selected cells."""
        selected_data = []
        for i in range(len(self.selected_indices)):
            selected_index = self.selected_indices[i]
            experiment_data: RawData
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

    def add_primary_trace(self, data: RawData, color: str) -> None:
        """Add the primary trace to the plot.

        Args:
            data (RawData): The data to plot.
            color (str): The color for the trace.
        """
        plot_data = data
        if self.zero_x:
            canonical_col = BDF.lookup_by_quantity(self.x_quantity).name
            plot_data = data.zero_column(canonical_col)
        primary_trace = go.Scatter(
            x=plot_data.get(self.x_axis),
            y=plot_data.get(self.y_axis),
            mode="lines",
            name=f"{data.info.get(self.cell_identifier, '')}",
            line={"color": color},
        )
        self.fig.add_trace(primary_trace)

    def add_secondary_trace(self, data: RawData, color: str) -> None:
        """Add the secondary trace to the plot.

        Args:
            data (RawData): The data to plot.
            color (str): The color for the trace.
        """
        plot_data = data
        if self.zero_x:
            canonical_col = BDF.lookup_by_quantity(self.x_quantity).name
            plot_data = data.zero_column(canonical_col)
        secondary_trace = go.Scatter(
            x=plot_data.get(self.x_axis),
            y=plot_data.get(self.secondary_y_axis),
            mode="lines",
            name=f"{data.info.get(self.cell_identifier, '')}",
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

        # Get data first so we can resolve available columns
        selected_data = self.get_data()
        self.selected_data = selected_data  # Store for use in property methods

        ax_col1, ax_col2, ax_col3, ax_col4 = st.columns(4)
        self.x_quantity = ax_col1.selectbox("x quantity", self.x_quantity_options)
        self.x_unit = ax_col2.selectbox(
            "x unit", self._quantity_unit_options[self.x_quantity]
        )
        self.zero_x = ax_col2.checkbox("Zero x")
        self.y_quantity = ax_col3.selectbox(
            "y quantity", self.y_quantity_options, index=0
        )
        self.y_unit = ax_col4.selectbox(
            "y unit", self._quantity_unit_options[self.y_quantity]
        )

        sec_col1, sec_col2, sec_col3 = st.columns(3)
        sec_y_options = ["None"] + self.y_quantity_options
        self.secondary_y_quantity = sec_col1.selectbox(
            "Secondary y quantity", sec_y_options
        )
        sec_unit_opts = self._quantity_unit_options.get(self.secondary_y_quantity, [""])
        self.secondary_y_unit = sec_col2.selectbox(
            "Secondary y unit",
            sec_unit_opts,
            key=f"sec_unit_{self.secondary_y_quantity}",
        )
        self.cell_identifier = sec_col3.selectbox(
            "Legend label",
            self.info.collect_schema().names(),
        )

        selected_names: list[str] = [
            str(getattr(self.cell_list[i], "info", {}).get(self.cell_identifier, ""))
            for i in self.selected_indices
        ]
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
                theme="streamlit",
            )

        if selected_data:
            tabs = st.tabs(selected_names)
            for tab_idx, tab in enumerate(tabs):
                # Resolve only columns that exist in this dataset
                data = selected_data[tab_idx]
                available_columns = self._resolve_available_columns(
                    data, self._display_columns_all
                )
                if available_columns:
                    resolved_exprs = [
                        data.columns.resolve(col) for col in available_columns
                    ]
                    tab.dataframe(
                        data.data.select(resolved_exprs).to_pandas(),
                        hide_index=True,
                    )
                else:
                    tab.warning(
                        "No display columns available in this dataset. "
                        "Available columns: " + ", ".join(data.columns.names)
                    )


if __name__ == "__main__":
    with open("dashboard_data.pkl", "rb") as f:
        cell_list = pickle.load(f)
    _Dashboard(cell_list).run()
