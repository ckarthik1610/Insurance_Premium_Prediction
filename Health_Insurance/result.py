from Health_Insurance.preprocessing import preprocess
import joblib
import pandas as pd

class result(preprocess):
    def  __init__(self, age, sex, bmi, children, smoker, region, path="Health_Insurance/random_forest.pkl"):
        preprocess.__init__(self,path)
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def predict(self):
        data = pd.DataFrame([{
            "age": self.age,
            "sex": self.sex,
            "bmi": self.bmi,
            "children": self.children,
            "smoker": self.smoker,
            "region": self.region
        }])
        
        model = joblib.load(self.file_directory)

        prediction = model.predict(data)

        return prediction
