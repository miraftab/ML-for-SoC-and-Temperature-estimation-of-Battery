import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
