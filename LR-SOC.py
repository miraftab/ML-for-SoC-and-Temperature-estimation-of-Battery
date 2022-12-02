import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

# from utilities import GridSearchCV_lr, Use this function to search and find best model for linear regression
from utilities import DistributionPlot


# a dictionary to store model scores
models_scores = {
    "model_names": [],
    "poly_degree": [],
    "r2": [],
    "mae": [],
    "rmse": [],
    "train_r2": [],
    "test_r2": [],
    "train_mae": [],
    "test_mae": [],
    "train_rmse": [],
    "test_rmse": [],
}


# create polynomial linear model
def linear_model(df, model_name, degree):
    # Feature selection
    features = ["ElapsedTime", "Temperature", "Voltage", "Current"]
    X = df[features]
    y = df["SOC"]

    # Normalize data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42
    )

    # Linera Model
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    X_poly = poly.fit_transform(X_norm)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predicting results
    yhat_train = model.predict(X_train_poly)
    yhat_test = model.predict(X_test_poly)
    yhat = model.predict(X_poly)

    # update model score dictionary
    models_scores["model_names"].append(model_name)
    models_scores["poly_degree"].append(degree)
    models_scores["r2"].append(model.score(X_poly, y))
    models_scores["mae"].append(mean_absolute_error(y, yhat))
    models_scores["rmse"].append(mean_squared_error(y, yhat, squared=True))
    models_scores["train_r2"].append(model.score(X_train_poly, y_train))
    models_scores["test_r2"].append(model.score(X_test_poly, y_test))
    models_scores["train_mae"].append(mean_absolute_error(y_train, yhat_train))
    models_scores["test_mae"].append(mean_absolute_error(y_test, yhat_test))
    models_scores["train_rmse"].append(
        mean_squared_error(y_train, yhat_train, squared=True)
    )
    models_scores["test_rmse"].append(
        mean_squared_error(y_test, yhat_test, squared=False)
    )

    # return y, yhat, y_test, yhat_test, y_train, yhat_train
    return y, yhat


def main():
    # read data
    df = pd.read_csv("data/wd_all.csv", index_col="Unnamed: 0")

    # use linear_model function
    y_all, yhat_all = linear_model(df, "df_all", 5)

    df_score = pd.DataFrame(models_scores)
    print(df_score)

    # plot results
    DistributionPlot(
        y_all,
        yhat_all,
        "Actual Values",
        "Predicted Values",
        "SOC Prediction - Actual Values VS Predicted Values - My DataFrame",
    )


if __name__ == "__main__":
    main()
