import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

random_state = 5805
pd.set_option('display.max_columns', None)
file_path = r"C:\Users\alvin\Dropbox\Documents_VT\2025_Fall\Machine Learning\Homework6\Carseats.csv"
df = pd.read_csv(file_path, low_memory=False)
global selected_columns
global step_stats
global rf_stats


def fun1():
    # a.
    grouped = df.groupby(['ShelveLoc', 'US'])['Sales'].sum().unstack().round(3)
    print(grouped)

    grouped.plot(kind='barh')
    plt.xlabel('Sales', fontsize=12)
    plt.ylabel('Shelve Location', fontsize=12)
    plt.title('Total Sales by Shelve Location in US', fontsize=14)
    plt.grid(True, linewidth=0.5)
    plt.legend(title='US')
    plt.tight_layout()
    plt.show()

    max_sales = grouped.stack().idxmax()
    max_value = grouped.stack().max()
    print(f"best sales are: {max_sales[0]}"
          f", location: {'US' if max_sales[1] == 'Yes' else 'Non-US'}"
          f", sales: {max_value:.3f}")

    # b.
    qualitatives = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=qualitatives, drop_first=True)
    print("One hot encoded:")
    print(df_encoded.head(5).round(3))

    # c.
    train, test = train_test_split(df_encoded, test_size=0.2, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    encoded_cols = train.columns[train.nunique() == 2]
    numeric_cols = [col for col in train.columns if col not in encoded_cols]

    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])
    print("Train set:")
    print(train.head(5).round(3))
    print("Test set:")
    print(test.head(5).round(3))


def fun2():
    # a.
    x = df.drop(columns=['Sales'])
    x = pd.get_dummies(x, drop_first=True)
    x = x.astype('float32')
    x = sm.add_constant(x)

    y = df['Sales']

    current_x = x.copy()
    steps = []

    while True:
        model = sm.OLS(y, current_x).fit()

        pvalues = model.pvalues.drop("const", errors='ignore')
        max_p = pvalues.max()
        max_p_var = pvalues.idxmax()

        steps.append({
            'Removed Feature': max_p_var,
            'p-value': round(max_p, 3),
            'AIC': round(model.aic, 3),
            'BIC': round(model.bic, 3),
            'Adjusted R^2': round(model.rsquared_adj, 3)
        })

        if max_p <= 0.01:
            break

        current_x = current_x.drop(columns=[max_p_var])

    step_df = pd.DataFrame(steps)
    print(step_df)
    print("\nfinal selected features:")
    print(model.summary())

    # b.
    global selected_columns
    selected_columns = list(current_x.columns)
    selected_x = sm.add_constant(x[selected_columns])
    selected_x_train, selected_x_test, y_train, y_test = train_test_split(selected_x, y,
                                                                          test_size=0.2,
                                                                          random_state=5805,
                                                                          shuffle=True)

    final_model = sm.OLS(y_train, selected_x_train).fit()
    y_pred = final_model.predict(selected_x_test)

    plt.plot(range(len(y_test)), y_test.values, label='Original Sales')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Sales', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Sales')
    plt.title('Original vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # c.
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error(MSE): {mse:.3f}")

    # for fun5
    global step_stats
    step_stats = {
        'Method': 'Stepwise Regression',
        'R-squared': final_model.rsquared,
        'Adjusted R-squared': final_model.rsquared_adj,
        'AIC': final_model.aic,
        'BIC': final_model.bic,
        'MSE': mse
    }
    # for fun6
    global final_model6, selected_x_test6, y_test6
    final_model6 = final_model
    selected_x_test6 = selected_x_test
    y_test6 = y_test


def fun3():
    # a.
    x = df.drop(columns=['Sales'])
    x = pd.get_dummies(x, drop_first=True)
    x = x.astype('float32')

    pca = PCA()
    pca.fit(x)

    explained_cumsum = np.cumsum(pca.explained_variance_ratio_)
    # ex. [0.42, 0.63, 0.77, 0.89, 1.00]
    feature_95 = np.argmax(explained_cumsum >= 0.95) + 1
    print(f'features needed to explain >95% variance: {feature_95}')

    # b.
    plt.plot(range(1, len(explained_cumsum) + 1), explained_cumsum, marker='o')
    plt.xlabel('Number of features')
    plt.ylabel('Cumulative explained variance')
    plt.title('CEV verses number of features')
    plt.grid(True)
    plt.show()

    # c.
    plt.plot(range(1, len(explained_cumsum) + 1), explained_cumsum, marker='o')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95%')
    plt.axvline(x=feature_95, color='green', linestyle='--', label=f'{feature_95} features')
    plt.xlabel('Number of features')
    plt.ylabel('Cumulative explained variance')
    plt.title('CEV verses number of features')
    plt.legend()
    plt.grid(True)
    plt.show()


def fun4():
    # a.
    x = df.drop(columns=['Sales'])
    x = pd.get_dummies(x, drop_first=True)
    x = x.astype('float32')
    y = df['Sales']

    rf = RandomForestRegressor(random_state=5805)
    rf.fit(x, y)
    importances = rf.feature_importances_
    feat_labels = x.columns

    indices = importances.argsort()[::-1]
    sorted_features = [feat_labels[i] for i in indices]
    sorted_importances = importances[indices]

    plt.barh(sorted_features, sorted_importances)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # b.
    threshold = 0.01
    selected = [name for name, imp in zip(feat_labels, importances) if imp >= threshold]
    eliminated = [name for name, imp in zip(feat_labels, importances) if imp < threshold]
    print("Eliminated:", eliminated)
    print("Selected:", selected)
    global selected_columns
    print("Random Forest verses Stepwise same:", set(selected) == set(selected_columns))

    # c.
    x_train, x_test, y_train, y_test = train_test_split(x[selected], y,
                                                        test_size=0.2,
                                                        random_state=5805,
                                                        shuffle=True)
    x_train = sm.add_constant(x_train)
    x_test = sm.add_constant(x_test)
    model = sm.OLS(y_train, x_train).fit()
    y_pred = model.predict(x_test)

    plt.plot(range(len(y_test)), y_test.values, label='Original Sales')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Sales', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Sales')
    plt.title('Original vs Predicted Sales(RF)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # e.
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error(MSE): {mse:.3f}")

    # for fun5
    global rf_stats
    rf_stats = {
        'Method': 'Random Forest Selection',
        'R-squared': model.rsquared,
        'Adjusted R-squared': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic,
        'MSE': mse
    }


def fun5():
    global step_stats
    global rf_stats
    compare = pd.DataFrame([step_stats, rf_stats])
    print(compare.round(3))


def fun6():
    global final_model6, selected_x_test6, y_test6
    pred = final_model6.get_prediction(selected_x_test6)
    pred_interval = pred.summary_frame()

    y_pred = pred_interval['mean']
    lower = pred_interval['obs_ci_lower']
    upper = pred_interval['obs_ci_upper']

    plt.plot(range(len(y_pred)), y_pred, label='predicted sales')
    plt.plot(range(len(lower)), lower, label='lower 95% interval', linestyle='--')
    plt.plot(range(len(upper)), upper, label='upper 95% interval', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Sales')
    plt.title('predicted sales value(with 95% interval)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fun7():
    # a.
    x = df[['Price']]
    y = df['Sales']
    param = {'polynomialfeatures__degree': range(1, 15)}
    model = make_pipeline(PolynomialFeatures(), LinearRegression())
    grid = GridSearchCV(model, param_grid=param, scoring='neg_root_mean_squared_error')
    grid.fit(x, y)

    print(f"minimized RMSE: {-grid.best_score_:.3f}")

    # b.
    optimal_n = grid.best_params_['polynomialfeatures__degree']
    print(f"optimal order n: {optimal_n}")

    # c.
    result = pd.DataFrame(grid.cv_results_)
    plt.plot(-result['mean_test_score'], result['param_polynomialfeatures__degree'])
    plt.xlabel('RMSE')
    plt.ylabel('n order')
    plt.title('RMSE versus the n order')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # d.
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=5805,
                                                        shuffle=True)
    model_best = make_pipeline(PolynomialFeatures(degree=optimal_n), LinearRegression())
    model_best.fit(x_train, y_train)
    y_pred = model_best.predict(x_test)

    plt.plot(range(len(y_test.values)), y_test, label='Actual Sales')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Sales', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Sales')
    plt.title('test set (sales) vs predicted sales values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # e.
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error(MSE) of optimal_n({optimal_n}): {mse:.3f}")


# fun1()
fun2()
# fun3()
# fun4()
# fun5()
# fun6()
# fun7()
