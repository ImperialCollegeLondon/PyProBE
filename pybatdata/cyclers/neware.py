import os
import re
import polars as pl

class Neware:
    @staticmethod
    def load_file(filepath):
        file_ext = os.path.splitext(filepath)[1]
        if file_ext == '.xlsx':
            df = pl.read_excel(filepath, engine='calamine').lazy()
        elif file_ext == '.csv':
            df = pl.scan_csv(filepath)
        column_dict = {'Date': 'Date',
                       'Cycle Index': 'Cycle',
                       'Step Index': 'Step',
                       'Current(A)': 'Current (A)',
                       'Voltage(V)': 'Voltage (V)',
                       'DChg. Cap.(Ah)': 'Discharge Capacity (Ah)',
                       'Chg. Cap.(Ah)': 'Charge Capacity (Ah)',
                       }
        df = Neware.convert_units(df)
        df = df.select(list(column_dict.keys())).rename(column_dict)
        df = df.with_columns(pl.col('Charge Capacity (Ah)').diff().alias('dQ_charge'))
        df = df.with_columns(pl.col('Discharge Capacity (Ah)').diff().alias('dQ_discharge'))
        df = df.with_columns(pl.col('dQ_charge').clip(lower_bound=0).fill_null(strategy="zero"))
        df = df.with_columns(pl.col('dQ_discharge').clip(lower_bound=0).fill_null(strategy="zero"))
        df = df.with_columns(((pl.col('dQ_charge')-pl.col('dQ_discharge')).cum_sum()
                              + pl.col('Charge Capacity (Ah)').max()).alias('Capacity (Ah)'))
        if df.dtypes[df.columns.index('Date')] != pl.Datetime:
            df = df.with_columns(pl.col('Date').str.to_datetime().alias('Date'))
        df = df.with_columns(pl.col('Date').dt.timestamp('ms').alias('Time (s)'))
        df = df.with_columns(pl.col('Time (s)') - pl.col('Time (s)').first())
        df = df.with_columns(pl.col('Time (s)')*1e-3)
        df = df.select(['Date',
                       'Time (s)',
                       'Cycle',
                       'Step',
                       'Current (A)',
                       'Voltage (V)',
                       'Capacity (Ah)',
                       ])
        return df

    @staticmethod
    def convert_units(df):
        conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}
        for column in df.columns:
            match = re.search(r'\((.*?)\)', column)
            if match:
                unit = match.group(1)
                prefix = next((x for x in unit if not x.isupper()), None)
                if prefix in conversion_dict:
                    df = df.with_columns((pl.col(column) * conversion_dict[prefix]).alias(column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')))
                    #df = df.with_columns(column, df[column] * conversion_dict[prefix])
                    #df = df.rename(column, column.replace('('+unit+')', '('+unit.replace(prefix, '')+')'))
        return df
