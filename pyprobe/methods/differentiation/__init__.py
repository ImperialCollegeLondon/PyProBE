"""A module for differentiation methods."""
from pyprobe.methods.differentiation import LEAN
from pyprobe.methods.methodregistry import MethodRegistry

gradient = MethodRegistry()
gradient.register_method("LEAN", LEAN.DifferentiateLEAN)
