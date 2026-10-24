import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame


def train():
    df = load_data()

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    print(f"Linear Regression RMSE: {lr_rmse:.4f}")

    # Model 2: Decision Tree
    dt = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_preds))
    print(f"Decision Tree RMSE: {dt_rmse:.4f}")

    # Pick the winner
    if lr_rmse < dt_rmse:
        best_model = lr
        best_name = "LinearRegression"
        best_rmse = lr_rmse
    else:
        best_model = dt
        best_name = "DecisionTreeRegressor"
        best_rmse = dt_rmse

    print(f"Best model: {best_name} with RMSE {best_rmse:.4f}")

    joblib.dump(best_model, "model.joblib")
    print("Model saved as model.joblib")


if __name__ == "__main__":
    train()