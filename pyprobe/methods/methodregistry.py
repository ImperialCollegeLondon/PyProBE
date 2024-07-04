"""Module for registering methods to be used in the pyprobe package."""
from typing import Any, Callable, Dict, List, Type

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.result import Result


class MethodRegistry:
    """A class for registering methods to be used in the pyprobe package.

    Attributes:
        methods (Dict[str, Any]): A dictionary of methods.
            Format: {method_name: method_class}
    """

    methods: Dict[str, Callable[[Result, Any], BaseMethod]] = {}

    def __call__(
        self, result: Result, method: str, *args: Any, **kwargs: Any
    ) -> BaseMethod:
        """Call a method from the registry.

        Args:
            result (Result): The input data to the method.
            method (str): The method name to use for the calculation.
            *args: The arguments to pass to the method.
            **kwargs: The keyword arguments to pass to the method.

        Returns:
            BaseMethod: The instantiated method object.
        """
        return self.methods[method](result, *args, **kwargs)

    @classmethod
    def register_method(
        cls, name: str
    ) -> Callable[[Type[BaseMethod]], Type[BaseMethod]]:
        """Register a method with the registry.

        Args:
            name (str): The name of the method.

        Returns:
            Callable: A decorator function to register the method.
        """

        def decorator(method_cls: Type[BaseMethod]) -> Type[BaseMethod]:
            cls.methods[name] = method_cls
            return method_cls

        return decorator

    @classmethod
    def list_methods(cls) -> List[str]:
        """List the available methods in the registry.

        Returns:
            List[str]: A list of the available methods.
        """
        return list(cls.methods.keys())