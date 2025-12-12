import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

class data:

    CATEGORICAL_MAPS = {
        "Claim_3_Years": {"Yes": 1, "No": 0},
        "Owner_Employment_Status": {"Yes": 1, "No": 0},
        "Accidental_Damage": {"Yes": 1, "No": 0},
        "Owner_Sex": {"M": 1, "F": 0},
        "Alarm_Present": {"Yes": 1, "No": 0},
        "Locks_Present": {"Yes": 1, "No": 0},
        "Flooding": {"Yes": 1, "No": 0},
        "Safe_Installed": {"Yes": 1, "No": 0}
    }

    def __init__(self, dir):
        self.path = dir

    def encoding(self,df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Normalize and map categorical columns
        for col in [
            "Claim_3_Years",
            "Owner_Employment_Status",
            "Accidental_Damage",
            "Alarm_Present",
            "Locks_Present",
            "Flooding",
            "Safe_Installed",
        ]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()          # "yes", "Yes", "YES" -> "YES"
                )
                df[col] = df[col].map({"YES": 1, "NO": 0})
                df[col] = df[col].fillna(0)  # just in case something weird appears

        # Owner_Sex: M/F
        if "Owner_Sex" in df.columns:
            df["Owner_Sex"] = (
                df["Owner_Sex"]
                .astype(str)
                .str.strip()
                .str.upper()              # "m", "M" -> "M"
            )
            df["Owner_Sex"] = df["Owner_Sex"].map({"M": 1, "F": 0})
            df["Owner_Sex"] = df["Owner_Sex"].fillna(0)

        # Handle numeric NaNs with median (Bedrooms, YearBuilt, premium if any missing)
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def preprocess(self):
        self.data = pd.read_csv(self.path)

        encoded = self.encoding(self.data)
        self.predictors = encoded.drop("Annual_Premium_Price", axis=1)
        self.response = encoded["Annual_Premium_Price"]
        
        self.X = self.predictors.astype(float)
        self.features = self.X.columns.tolist()
        
    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.response)

    def save(self):
        with open("Home_Insurance/Linear_Regression.pkl","wb") as f:
            pickle.dump(self.model, f)

        with open("Home_Insurance/Feature_names.pkl","wb") as f:
            pickle.dump(self.features, f)

if __name__ == "__main__":
    x = data("Home_Insurance/dataset.csv")
    x.preprocess()
    x.train()
    x.save()

