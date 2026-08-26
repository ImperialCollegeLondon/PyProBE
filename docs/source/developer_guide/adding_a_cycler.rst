Adding a Cycler
===============

PyProBE owns no cycler reader of its own. Every raw file, whatever the vendor,
reads through :func:`bdf.io.scan`, which resolves a plugin from the
``battery-data-format`` package (imported as ``bdf``) and normalises the file's
columns to the BDF ontology before :meth:`~pyprobe.filters.Procedure.load` ever
sees it.

Where auto-detection looks
--------------------------
``bdf`` keeps two registries, and both are plain dictionaries declared inside the
``bdf`` package itself:

- ``bdf.table_normalizers.NORMALIZERS`` maps a vendor name to a
  ``TableNormalizer``, which relates the vendor's column headings to the BDF
  ontology.
- ``bdf.plugins.PLUGINS`` maps a plugin id, such as ``"neware_csv"`` or
  ``"biologic_mpt"``, to a ``Plugin``, which pairs a table parser (carrying its
  normalizer) with a metadata parser. ``bdf.plugins.detect`` chooses a
  candidate from this registry by file extension, magic bytes, embedded
  metadata, and finally column names, and :func:`bdf.io.scan` calls
  ``detect`` whenever its own ``plugin`` argument is left at its default.

Building a plugin declaratively
-------------------------------
A caller can build a working plugin without a change to ``bdf`` at all.
:func:`bdf.plugins.dump_plugins` writes an existing plugin to a JSON or a
YAML file, so a reader gets a working template to edit rather than a blank
page. :func:`bdf.plugins.load_plugins` reads such a file back as a
``{id: Plugin}`` dict, and :meth:`~pyprobe.filters.Procedure.load` takes a
loaded plugin under its ``plugin`` argument.

The following commands dump the built-in Arbin CSV plugin, edit the dumped
file so it reads a differently-named current column, and read a file that
carries that renamed column:

.. code-block:: python

   import bdf.plugins as p

   p.dump_plugins({"my_arbin": p.PLUGINS["arbin_csv"]}, "my_arbin.yaml")

The dumped file holds the column-header rules directly, one entry per BDF
quantity:

.. code-block:: yaml

   my_arbin:
     table_parser:
       normalizer:
         current_ampere:
         - hdr: Current ({unit})
           assumed: false
           legacy: false
           reverse_sign: false
         ...

Adding an ``hdr`` entry under ``current_ampere``, such as ``Amps ({unit})``,
teaches the plugin to read a file whose current column is headed
``"Amps (A)"`` instead of ``"Current (A)"``. Loading the edited file back and
passing it to :meth:`~pyprobe.filters.Procedure.load` reads the renamed
column into ``Current / A``:

.. code-block:: python

   from pyprobe.filters import Procedure

   loaded = p.load_plugins("my_arbin.yaml")
   procedure = Procedure.load("path/to/file.csv", plugin=loaded["my_arbin"])

Without the added ``hdr`` entry, the same call raises
:class:`bdf._errors.BDFValidationError`, naming ``Current / A`` among the
missing required columns, because none of the plugin's other headers match
the renamed one.

What the declarative route does not give
----------------------------------------
:func:`bdf.io.scan` auto-detects only from its own ``bdf.plugins.PLUGINS``
registry. A plugin that a caller builds this way is never a candidate for
that detection, so it must be passed under ``plugin=`` on every
:meth:`~pyprobe.filters.Procedure.load` call that reads a file in that
format. A cycler that auto-detection must find, without a caller naming a
plugin, still belongs upstream: contributing a ``TableNormalizer`` and a
``Plugin`` to ``battery-data-format`` itself, not writing a module inside
PyProBE.

.. footbibliography::
