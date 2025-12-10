import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/stock%20prices.csv"
stock_df = pd.read_csv(url)
print(stock_df.head())


#2.
def fun2 ():
    unique_companies = stock_df['symbol'].unique()
    print("number of unique companies:", len(unique_companies))
    print("companies:", unique_companies)

    google_apple = stock_df[(stock_df['symbol'] == "GOOGL")| (stock_df['symbol'] == "AAPL")]
    google_apple['date'] = pd.to_datetime(google_apple['date'], format='%Y-%m-%d')
    google_apple = google_apple.sort_values('date')

    for sym, gr in google_apple.groupby('symbol'):
        plt.plot(gr['date'], gr['close'], label=sym)

    plt.title("Closing Stock Value of Google and Apple")
    plt.xlabel("Date")
    plt.ylabel("Closing Value")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha ='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def fun3():
    aggregated = stock_df.groupby('symbol').sum(numeric_only=True)
    print("number of records in the cleaned dataset:", len(stock_df))
    print("number of records in the aggregated dataset:", len(aggregated))
    print("the first five rows of the aggregated dataset:")
    print(aggregated.head())

def fun4():
    sliced = stock_df[["symbol", "close", "volume"]]
    aggregated = sliced.groupby('symbol').agg(["mean", "var"])

    close_var = aggregated[("close", "var")].values
    syms = aggregated.index.values

    idx =np.argmax(close_var)
    sym = syms[idx]
    maxvalue = np.max(close_var)
    print("the company with the maximum variance in closing price:", sym)
    print(f"variance value: {maxvalue:.2f}")

def fun5():
    stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y-%m-%d')
    Google = stock_df[(stock_df['symbol'] == 'GOOGL') & (stock_df['date'] >="2015-01-01")][["date", "close"]]
    print("first five rows of Google’s closing stock price\n", Google.head())
    return Google

def fun6(Google):
    g_rolling = Google['close'].rolling(window=30, center=True).mean()

    plt.figure(figsize = (12,8))
    plt.plot(Google['date'], Google['close'], label="Original price", alpha= 0.5)
    plt.plot(Google['date'], g_rolling, label="30-day window", color="red", linewidth=2)

    plt.title("30-days Closing price rolling mean of Google")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    excluded = g_rolling.isna().sum()
    print("the number of excluded:", excluded)

def fun7(Google):
    labels = ("verylow", "low", "normal", "high", "very high")
    Google["price_catagory"] = pd.cut (Google["close"], bins = 5, labels = labels)
    print("Google.head(10):\n", Google.head(10))

    plt.figure(figsize = (12,8))
    sns.countplot(x = "price_catagory", data = Google, order = labels)
    plt.title("Count of Goolge closing price categories")
    plt.xlabel("price_catagory")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
    return Google

def fun8(Google):
    plt.figure(figsize = (12,8))
    plt.hist(Google["close"], bins = 5, edgecolor = "black")
    plt.title("Histogram of Google closing price")
    plt.xlabel("Closing Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def fun9(Google):
    labels = ("verylow", "low", "normal", "high", "very high")
    Google["price_category"] = pd.qcut(Google["close"], q = 5, labels = labels)
    print("Google.head(10):\n", Google.head(10))

    plt.figure(figsize = (12,8))
    sns.countplot(x = "price_category", data = Google, order = labels)
    plt.title("Count of Google closing price categories (Frequency)")
    plt.xlabel("price_category")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

def fun10():
    stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y-%m-%d')
    features = ["open", "high", "low", "close", "volume"]
    Google = stock_df[(stock_df['symbol'] == 'GOOGL') & (stock_df['date'] >= "2015-01-01")][features]
    data = Google.values

    means = np.mean(data, axis=0)
    centered = data - means
    cov_matrix = (centered.T @ centered) / (Google.shape[0] - 1)
    print("Manually computed Covariance Matrix:\n", pd.DataFrame(cov_matrix, index=features, columns=features))

def fun11():
    stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y-%m-%d')
    features = ["open", "high", "low", "close", "volume"]
    Google = stock_df[(stock_df['symbol'] == 'GOOGL') & (stock_df['date'] >= "2015-01-01")][features]

    cov = Google[features].cov()
    print(".cov() Covariance Matrix :\n", cov)


#1.
missing_values = stock_df.isnull().sum()
missing_features = missing_values[missing_values >0]
print("Missing features:\n", missing_features)

stock_df.fillna(stock_df.mean(numeric_only= True), inplace=True)
print("fillna completed")

remaining_missing = stock_df.isnull().sum().sum()
if(remaining_missing == 0):
    print("All missing values have been replaced.")
else: print("There are {} missing values".format(remaining_missing))

choice = input("Enter your choice: ")
if choice == "2":
    fun2()
elif choice == "3":
    fun3()
elif choice == "4":
    fun4()
elif choice == "5":
    fun5()
elif choice == "6":
    g = fun5()
    fun6(g)
elif choice == "7":
    g2 = fun5()
    fun7(g2)
elif choice == "8":
    g3= fun5()
    g4 = fun7(g3)
    fun8(g4)
elif choice == "9":
    g5 = fun5()
    fun9(g5)
elif choice == "10":
    fun10()
elif choice == "11":
    fun11()
elif choice == "0":
    print("Goodbye!")
else:
    print("Invalid choice. Please try again.")
