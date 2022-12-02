import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split

mpl.style.use("ggplot")

# Plot function for results
def DistributionPlot(redFunc, blueFunc, redName, blueName, title):
    width, height = 12, 10

    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(
        redFunc,
        color="r",
        lw=3,
        label=redName,
    )
    ax2 = sns.kdeplot(blueFunc, color="b", label=blueName, ax=ax1)

    plt.title(title)
    plt.legend()

    plt.show()


# create polynomial linear regression pipline
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# GridsearchCV function to select best polynomial degree
def GridSearchCV_lr(df, features, target_feature):
    # Feature selection
    X = df[features]
    y = df[target_feature]

    # Normalize data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.25, random_state=42
    )

    # GridsearchCV
    params = [{"polynomialfeatures__degree": np.arange(1, 6)}]
    grid = GridSearchCV(PolynomialRegression(), param_grid=params, cv=5)
    grid.fit(X_train, y_train)

    # Select best model and degree
    model = grid.best_estimator_
    degree = model.get_params()["polynomialfeatures__degree"]

    return model, degree
