.. _the_test_protocol:

The Test Protocol
=================

PyProBE holds the experiment definitions of a procedure as a test protocol tree,
under ``metadata.battinfo_test_protocol.method``. Every filter that selects an
experiment, a cycle, or a step reads that tree, and a filtered object reports the
subtree that produced it.

The step tree
-------------
The protocol is an ordered list of :class:`~pyprobe.protocol.Step` records. Each
record is either a group or a leaf:

- A **group** carries ``mode="group"`` and a ``description``, and it holds a list
  of child steps under ``steps``. A group names an experiment where its
  ``description`` is set, and it repeats where its ``count`` is set. A group can do
  both, and a group can hold a further group, so an experiment can contain a
  further experiment.
- A **leaf** carries a ``description`` of the instruction the cycler ran, and it
  holds no children.

The tag convention
------------------
:class:`~pyprobe.protocol.Step` declares no field for the cycler's own step
number, so a leaf carries it in its ``tags`` list, as a string of the form
``"step_id:<n>"``. :func:`~pyprobe.protocol.step_id_of` reads the first such tag
of a node, and :func:`~pyprobe.protocol.step_id_tag` writes one:

.. code-block:: python

   from pyprobe.protocol import step_id_of, step_id_tag

   step_id_tag(4)
   # Returns: 'step_id:4'

A leaf that carries no ``"step_id:"`` tag resolves to no cycler step, so a filter
that needs one raises where every leaf of the group it selects carries none.

The derived views
-----------------
A procedure, an experiment, and a cycle each read the protocol tree through three
properties, rather than through a stored structure.

``experiment_names`` returns the descriptions of the groups directly below the
current tree level:

.. code-block:: python

   from pyprobe.filters import Procedure

   procedure = Procedure.load("tests/sample_data/neware/sample_data_neware.bdf.parquet")
   procedure.attach_legacy_readme("tests/sample_data/neware/README.yaml")

   procedure.experiment_names
   # Returns: ['Initial Charge', 'Break-in Cycles', 'Discharge Pulses']

``step_descriptions`` pairs each leaf's cycler step identifier with its
description, in tree order:

.. code-block:: python

   procedure.experiment("Initial Charge").step_descriptions
   # Returns: {'Step': [1, 2, 3],
   #           'Description': ['Rest for 4 hours',
   #                            'Charge at 4mA until 4.2 V, Hold at 4.2 V until 0.04 A',
   #                            'Rest for 2 hours']}

``cycle_info`` lists the bounds and the repeat count of every group below the
current tree level that carries a ``count``, as ``(start, end, count)`` tuples of
step identifiers:

.. code-block:: python

   procedure.experiment("Break-in Cycles").cycle_info
   # Returns: [(4, 7, 5)]

When a step index is exact
--------------------------
A ``step()`` call takes a positional index that counts step events in the data,
not leaves in the tree. Where no group at or before that index repeats, the index
addresses its leaf exactly, and the filtered result holds that leaf as its
protocol:

.. code-block:: python

   initial_charge = procedure.experiment("Initial Charge")
   initial_charge.step(0).step_descriptions
   # Returns: {'Step': [1], 'Description': ['Rest for 4 hours']}

Where a repeating group lies between the index and the end it counts from, one
repeat gives many data events for the leaves inside it, and every index past that
group shifts. A ``step()`` call across that shift keeps the protocol node of its
source rather than report the wrong leaf:

.. code-block:: python

   pulses = procedure.experiment("Discharge Pulses")
   pulses.cycle_info
   # Returns: [(9, 12, 10)]

   first_pulse_step = pulses.step(0)
   first_pulse_step.data.height
   # Returns: 101, the rows of the first repeat's first step alone

   first_pulse_step.step_descriptions
   # Returns the four leaves of the whole 'Discharge Pulses' group, because the
   # group above the leaf repeats

The filtered data is exact either way. Only the protocol that a filter reports
can be less precise than the data it selected, and only where a repeat lies on
the side of the index that counts toward it. A non-negative index is exact where
no group at or before its leaf repeats. A negative index counts from the end, so
it is exact where no group at or after its leaf repeats.

A leaf that itself repeats no group, but that follows a repeating group earlier
in the tree, is shifted by that earlier repeat as well. The rule looks at every
group between the index and the end it counts from, not at the leaf's own group
alone.

A ``cycle()`` filter, an ``experiment()`` filter, and a ``step()`` filter each
reduce the tree this way, because each names its target node before it reads the
data. A condition filter, such as ``charge()``, ``discharge()``, ``rest()``, or
``constant_current()``, builds its mask from the data instead, so it keeps the
protocol node of its source and performs no reduction of its own.

Attaching a protocol from a README
----------------------------------
A source that carries no protocol of its own can attach one from a legacy
``README.yaml`` file. See the :ref:`writing_a_readme_file` guide.

.. footbibliography::
