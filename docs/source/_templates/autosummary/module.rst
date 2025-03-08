{{ fullname | escape | underline}}

{% if fullname == "pyprobe" %}
.. admonition:: Top-Level Imports
   :class: note

   The following items can be imported directly from the ``pyprobe`` package:

   **Classes:**
   
   * :py:class:`pyprobe.Cell <pyprobe.cell.Cell>`
   * :py:class:`pyprobe.Result <pyprobe.result.Result>`
   
   **Functions:**
   
   * :py:func:`pyprobe.load_archive <pyprobe.cell.load_archive>`
   * :py:func:`pyprobe.make_cell_list <pyprobe.cell.make_cell_list>`
   * :py:func:`pyprobe.process_cycler_data <pyprobe.cell.process_cycler_data>`
   * :py:func:`pyprobe.launch_dashboard <pyprobe.dashboard.launch_dashboard>`
{% endif %}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

.. footbibliography::