"""A module for the test protocol tree that PyProBE holds in its metadata.

PyProBE stores the experiment definitions of a procedure in
``metadata.battinfo_test_protocol.method``, an ordered list of BDF
:class:`Step` records. A group node carries ``mode="group"`` and a
``description``, and it repeats when it carries a ``count``. A leaf node
carries a ``description`` and its cycler step identifier.

:class:`Step` declares no step identifier field, so a leaf carries that
identifier in ``tags`` as ``"step_id:<n>"``. This module owns the parse of
that convention and the walk over the tree.

This module is the single import site for :class:`Step`. Every other PyProBE
module takes the class from here.
"""

from bdf.battinfo.generated.test_protocol_schema import Step

__all__ = ["Step", "leaves", "step_id_of"]


def step_id_of(step: Step) -> int | None:
    """Return the cycler step identifier that a node carries.

    The identifier lives in the first ``"step_id:"`` tag of the node.

    Args:
        step: The protocol node to read.

    Returns:
        int | None: The step identifier, or None where the node carries no
            ``"step_id:"`` tag.

    Raises:
        ValueError: If the tag holds a value that is not an integer. The
            message names the description of the node.
    """
    raise NotImplementedError


def leaves(step: Step) -> list[Step]:
    """Return the leaf nodes under a protocol node, in tree order.

    A leaf is a node that holds no child steps. The node itself is a leaf
    where it holds none.

    Args:
        step: The protocol node to walk.

    Returns:
        list[Step]: The leaves under the node, in tree order.
    """
    raise NotImplementedError
