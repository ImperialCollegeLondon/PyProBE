"""A module to load and process Neware battery cycler data."""
import os
import re
import polars as pl

class Neware:
    """A Neware battery cycler object."""
    @staticmethod
    def load_file(filepath: str) -> pl.LazyFrame:
        """Load a Neware battery cycler file into PyBatData format.
        
        Args:
            filepath: The path to the file.
            
        Returns:
            A LazyFrame containing the data in PyBatData format.
        """
        file_ext = os.path.splitext(filepath)[1]
        if file_ext == '.xlsx':
            lf = pl.read_excel(filepath, engine='calamine').lazy()
        elif file_ext == '.csv':
            lf = pl.scan_csv(filepath)
        column_dict = {'Date': 'Date',
                       'Cycle Index': 'Cycle',
                       'Step Index': 'Step',
                       'Current(A)': 'Current [A]',
                       'Voltage(V)': 'Voltage [V]',
                       'DChg. Cap.(Ah)': 'Discharge Capacity [Ah]',
                       'Chg. Cap.(Ah)': 'Charge Capacity [Ah]',
                       }
        lf = Neware.convert_units(lf)
        lf = lf.select(list(column_dict.keys())).rename(column_dict)
        lf = lf.with_columns(pl.col('Charge Capacity [Ah]').diff().alias('dQ_charge'))
        lf = lf.with_columns(pl.col('Discharge Capacity [Ah]').diff().alias('dQ_discharge'))
        lf = lf.with_columns(pl.col('dQ_charge').clip(lower_bound=0).fill_null(strategy="zero"))
        lf = lf.with_columns(pl.col('dQ_discharge').clip(lower_bound=0).fill_null(strategy="zero"))
        lf = lf.with_columns(((pl.col('dQ_charge')-pl.col('dQ_discharge')).cum_sum()
                              + pl.col('Charge Capacity [Ah]').max()).alias('Capacity [Ah]'))
        if lf.dtypes[lf.columns.index('Date')] != pl.Datetime:
            lf = lf.with_columns(pl.col('Date').str.to_datetime().alias('Date'))
        lf = lf.with_columns(pl.col('Date').dt.timestamp('ms').alias('Time [s]'))
        lf = lf.with_columns(pl.col('Time [s]') - pl.col('Time [s]').first())
        lf = lf.with_columns(pl.col('Time [s]')*1e-3)
        lf = lf.select(['Date',
                       'Time [s]',
                       'Cycle',
                       'Step',
                       'Current [A]',
                       'Voltage [V]',
                       'Capacity [Ah]',
                       ])
        return lf

    @staticmethod
    def convert_units(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Convert units of a LazyFrame to SI.
        
        Args:
            lf: The LazyFrame to convert units of.
        
        Returns:
            The LazyFrame with converted units to SI.
        """
        conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}
        for column in lf.columns:
            match = re.search(r'\((.*?)\)', column)
            if match:
                unit = match.group(1)
                prefix = next((x for x in unit if not x.isupper()), None)
                if prefix in conversion_dict:
                    lf = lf.with_columns((pl.col(column) * conversion_dict[prefix]).alias(column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')))
        return lf
