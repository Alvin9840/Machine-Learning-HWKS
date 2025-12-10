import math
import seaborn as sns
import pandas as pd
from astropy.utils.metadata.utils import dtype
from fontTools.misc.classifyTools import classify
from pandas.core.interchange.dataframe_protocol import DataFrame
from pyarrow import nulls
from statsmodels.tools import categorical
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

# 1.
print("1. ")
available_datasets = sns.get_dataset_names()
print(available_datasets)


# 2.
# for classifying categorical and numerical
def classification(x):
    numerical_features = []
    categorical_features = []
    for column in x:
        if pd.api.types.is_numeric_dtype(x[column]):
            # prevent the 0/1 or boolean variables
            if x[column].nunique() <= 2:
                categorical_features.append(column)
            else:
                numerical_features.append(column)
        else:
            categorical_features.append(column)
    print("Categorical_features = ", categorical_features)
    print("Numerical_features = ", numerical_features)
    # to check if there is any exeption
    print(x.head())
    print(x.dtypes)
    print("\n")


print('\n'"2. ")
diamonds = pd.DataFrame(sns.load_dataset("diamonds"))
iris = pd.DataFrame(sns.load_dataset("iris"))
tips = pd.DataFrame(sns.load_dataset("tips"))
penguins = pd.DataFrame(sns.load_dataset("penguins"))
titanic = pd.DataFrame(sns.load_dataset("titanic"))

# for number of row
print("diamonds_shape = {} ".format(diamonds.shape))
classification(diamonds)

print("iris_shape = {} ".format(iris.shape))
classification(iris)

print("tips_shape = {} ".format(tips.shape))
classification(tips)

print("penguins_shape = {} ".format(penguins.shape))
classification(penguins)

print("titanic_shape = {} ".format(titanic.shape))
classification(titanic)

# 3.
print('\n'"3. ")
print("age: \n", titanic["age"].describe(), '\n')
print("sibsp: \n", titanic["sibsp"].describe(), '\n')
print("parch: \n", titanic["parch"].describe(), '\n')
print("fare: \n", titanic["fare"].describe(), '\n')


def missing(m):
    ms_count = 0
    for value in m:
        if pd.isnull(value):
            ms_count += 1

    print(ms_count)


for column in titanic:
    print("Number of missing values in ", column, ":")
    missing(titanic[column])

# 4.
print('\n'"4. ")
titanic_numerical = titanic[["age", "sibsp", "parch", "fare"]]
print("original dataset: \n", titanic.head())
print("numerical dataset: \n", titanic_numerical.head())

# 5.
print('\n'"5. ")
total_miss_counts = 0
total_entries = titanic_numerical.size
for column in titanic_numerical:
    miss_count = 0
    for value in titanic_numerical[column]:
        if pd.isnull(value):
            miss_count += 1
            total_miss_counts += 1
    print("missing observations of", column, ":", miss_count)

titanic_numerical = titanic_numerical.fillna(titanic_numerical.mean())

print("total_miss_counts: ", total_miss_counts)
s = total_miss_counts / total_entries * 100
print("percentage of each attribute’s entries were replaced during cleaning: ")
print(round(s, 2), "%")

# 8.
print('\n'"8. ")
# arithmetic_mean
arithmetic_mean = sum(titanic_numerical["age"]) / len(titanic_numerical["age"])
print("arithmetic_mean: ", round(arithmetic_mean, 2))

# geometric_mean
geometric_mean = 0
GM_log = 0
for value in titanic_numerical["age"]:
    GM_log += math.log(value)

geometric_mean = math.exp(GM_log / len(titanic_numerical["age"]))
print("geometric_mean: ", round(geometric_mean, 2))

# harmonic_mean
HM_sum = 0
for value in titanic_numerical["age"]:
    HM_sum += 1 / value

harmonic_mean = len(titanic_numerical["age"]) / HM_sum
print("harmonic_mean: ", round(harmonic_mean, 2))

#boxplot
plt.boxplot(titanic_numerical["age"])
plt.show()

#histograms
plt.hist(titanic_numerical["age"],bins=20, edgecolor="black")
plt.grid()
plt.xlabel('Age')
plt.ylabel('Amount')
plt.title('Histogram of age')
plt.show()

plt.hist(titanic_numerical["fare"],bins=20,color= "orange", edgecolor="black")
plt.grid()
plt.xlabel('Price')
plt.ylabel('Amount')
plt.title('Histogram of fare')
plt.show()

#pairwise bivariate distributions
PBD = sns.pairplot(titanic_numerical,diag_kind = "kde",kind = "scatter",)
PBD.fig.suptitle('Pairplot of titanic numerical',y=1)
for ax in PBD.axes.flat:
        ax.grid(True)
plt.grid()
plt.show()