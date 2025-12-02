import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import preprocess

def Training(n = 500):
    x = preprocess("insurance_data.csv")
    x.train_test(0.75)
    preprocessor = x.preprocessing()
    
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", RandomForestRegressor(
                n_estimators=n,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    # ===== Train Model =====
    model.fit(x.X_Train, x.Y_Train)

    Predictions = model.predict(x.X_Test)
    rmse = mean_squared_error(x.Y_Test, Predictions) ** 0.5
    r2 = r2_score(x.Y_Test, Predictions)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R^2 : {r2:.3f}")

    return model

def save(model,path):
    joblib.dump(model, path)

model = Training(500)
save(model,"random_forest.pkl")
