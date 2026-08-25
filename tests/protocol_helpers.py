"""Helpers for building a protocol tree in tests."""

from bdf import Metadata
from bdf.battinfo.generated.test_protocol_schema import BattinfoTestProtocol

from pyprobe.filters import Procedure
from pyprobe.protocol import Step


def attach_protocol(procedure: Procedure, method: list[Step]) -> None:
    """Attach a protocol tree to a procedure's metadata record.

    Args:
        procedure: The procedure to attach the tree to.
        method: The protocol tree, as an ordered list of top-level steps.
    """
    procedure.metadata = Metadata(
        battinfo_test_protocol=BattinfoTestProtocol(method=method),
    )
