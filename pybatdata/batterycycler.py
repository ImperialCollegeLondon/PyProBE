"""A module to load and process battery cycler data."""
import polars as pl

class BatteryCycler:
    """A battery cycler object."""
    @classmethod
    def load_file(cls, 
                  filepath: str) -> pl.LazyFrame:
        """Load a battery cycler file into PyBatData format.
        
        Args:
            filepath: The path to the file.
        """
        raise NotImplementedError

    @staticmethod
    def convert_units(lazyframe: pl.LazyFrame) -> pl.LazyFrame:
        """Convert units of a LazyFrame.
        
        Args:
            lazyframe: The LazyFrame to convert units of.
        """
        raise NotImplementedError