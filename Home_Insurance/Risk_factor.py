import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

class data:
    def __init__(self, dir):
        self.path = dir

    def preprocess(self):
        self.data = pd.read_csv(self.path)
        self.predictors = self.data.drop("Annual_Premium_Price", axis=1)
        self.response = self.data["Annual_Premium_Price"]
        
        self.X = pd.get_dummies(self.predictors,drop_first=True)
        self.features = self.X.columns
        
    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.response)

    def save(self):
        with open("Home_Insurance/Linear_Regression.pkl","wb") as f:
            pickle.dump(self.model, f)

        with open("Home_Insurance/Feature_names.pkl","wb") as f:
            pickle.dump(self.features, f)
