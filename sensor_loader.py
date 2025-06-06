import pandas as pd

class SensorDataLoader: 
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.sheet_names = []

    def load_sheet(self, sheet_index=0):
        xls = pd.read_excel(self.filepath, sheet_name=None, engine="openpyxl")
        self.sheet_names = list(xls.keys())[sheet_index]
        df = xls[self.sheet_names]
        time = df.iloc[:, 0]
        left = df.iloc[:, 1:9]
        right = df.iloc[:, 9:17]
        return time, left, right, self.sheet_names[sheet_index]