from omnisolver.common.plugin import Plugin

from omnisolver.pt import get_plugin


def test_plugin_can_be_loaded():
    assert isinstance(get_plugin(), Plugin)
