{{ fullname | escape | underline}}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {% set _path = item.split('.') %}
      {{ _path[-1] }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {% set _path = item.split('.') %}
      {{ _path[-1] }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
   {% for item in classes %}
      {% set _path = item.split('.') %}
      {{ _path[-1] }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {% set _path = item.split('.') %}
      {{ _path[-1] }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {% set _path = item.split('.') %}
   {{ _path[-1] }}
{%- endfor %}
{% endif %}
{% endblock %}
