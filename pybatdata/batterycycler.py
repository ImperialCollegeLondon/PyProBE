class BatteryCycler:
    @classmethod
    def load_file(cls, filepath):
        raise NotImplementedError

    @staticmethod
    def convert_units(df):
        raise NotImplementedError