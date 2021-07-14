import pandas as pd
import numpy as np

class ClassMapping:
    def __init__(self):
        self.mapping = pd.read_csv("classmap.csv")
                        
    def __call__(self, _id_):
        if np.isnan(_id_):
            return np.nan
        return self.get_catid(int(_id_))
                                        
    def get_catid(self, id_):
        return int(self.mapping[self.mapping["id"] == id_]["category_id"])
                                                        
    def reverse(self, category_id):
        return int(self.mapping[self.mapping["category_id"] == category_id]["id"])
