"""A module for the DMA method."""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.result import Result


class DMA(BaseMethod):
    """A method to calculate DMA parameters from fitted OCV curves."""

    def __init__(self, input_data: List[Result]) -> None:
        """Initialize the DMA method."""
        super().__init__(input_data)
        self.cell_capacity = self.variable("Cell Capacity")
        self.pe_capacity = self.variable("Cathode Capacity")
        self.ne_capacity = self.variable("Anode Capacity")
        self.li_inventory = self.variable("Li Inventory")
        SOH, LAM_pe, LAM_ne, LLI = self.calculate_dma_parameters(
            self.cell_capacity, self.pe_capacity, self.ne_capacity, self.li_inventory
        )

        self.dma_result = self.make_result(
            {
                "SOH": SOH,
                "LAM_pe": LAM_pe,
                "LAM_ne": LAM_ne,
                "LLI": LLI,
            }
        )
        self.dma_result.column_definitions = {
            "SOH": "Cell capacity normalized to initial capacity.",
            "LAM_pe": "Loss of active material in positive electrode.",
            "LAM_ne": "Loss of active material in positive electrode.",
            "LLI": "Loss of lithium inventory.",
        }
        self.output_data = self.dma_result

    @classmethod
    def calculate_dma_parameters(
        cls,
        cell_capacity: NDArray[np.float64],
        pe_capacity: NDArray[np.float64],
        ne_capacity: NDArray[np.float64],
        li_inventory: NDArray[np.float64],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Calculate the DMA parameters.

        Args:
            pe_stoich_limits (NDArray[np.float64]): The cathode stoichiometry limits.
            ne_stoich_limits (NDArray[np.float64]): The anode stoichiometry limits.
            pe_capacity (NDArray[np.float64]): The cathode capacity.
            ne_capacity (NDArray[np.float64]): The anode capacity.
            li_inventory (NDArray[np.float64]): The lithium inventory.

        Returns:
            Tuple[float, float, float, float]: The SOH, LAM_pe, LAM_ne, and LLI.
        """
        SOH = cell_capacity / cell_capacity[0]
        LAM_pe = 1 - pe_capacity / pe_capacity[0]
        LAM_ne = 1 - ne_capacity / ne_capacity[0]
        LLI = 1 - li_inventory / li_inventory[0]
        return SOH, LAM_pe, LAM_ne, LLI
