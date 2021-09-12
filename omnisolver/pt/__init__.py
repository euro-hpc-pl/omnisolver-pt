"""Implementation of parallel tempering plugin for Omnisolver."""
from omnisolver.plugin import Plugin, plugin_from_specification, plugin_impl
from pkg_resources import resource_stream
from yaml import safe_load


@plugin_impl
def get_plugin() -> Plugin:
    """Get package name and resource path."""
    specification = safe_load(resource_stream("omnisolver.pt", "pt.yml"))
    return plugin_from_specification(specification)
