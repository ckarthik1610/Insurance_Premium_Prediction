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
        self.file_directory = path

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

obj = result(age = 30,sex = "Male",bmi = 24,children = 1,smoker = "Yes",region = "Northwest",path = "Health_Insurance/Test_run.pkl")
prediction = obj.predict()