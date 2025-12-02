import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class preprocess:

    def __init__(self,path):
        self.file_directory = path
    
    def train_test(self,split=0.8):

        self.data = pd.read_csv(self.file_directory)
        self.X = self.data[["age", "sex", "bmi", "children", "smoker", "region"]]
        self.Y = self.data["charges"]

        self.X_Train,self.X_Test,self.Y_Train,self.Y_Test = train_test_split(
            self.X, self.Y, test_size=split, random_state=42
        )

    def preprocessing(self):
        self.numerical = ["age","children"]
        self.categorical = ["sex", "bmi", "smoker", "region"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical)
            ],
            remainder="passthrough"  # keep numeric features as they are
        )

        return self.preprocessor