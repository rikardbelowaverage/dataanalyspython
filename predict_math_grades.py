import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_data():
    data = pd.read_csv("student-mat.csv")
    return data


def process_data(data):
    data.replace({'GP': 0, 'MS': 1, 'F': 0, 'M': 1, 'LE3': 0, 'GT3': 1, 'T': 0, 'A': 1, 'U': 0, 'R': 1}, inplace=True)
    data.replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}, inplace=True)
    data.replace({'home': 1, 'reputation': 2, 'course': 3, 'other': 4, 'mother': 0, 'father': 1}, inplace=True)
    data.replace({'yes': 1, 'no': 0},inplace=True)
    y = data.G3
    x = data.drop(['G3'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print('Coefficients:', model.coef_)
    print(data.head())
    print('Intercept:', model.intercept_)
    print('Mean squared error (MSE): %.2f'
          % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination (R^2): %.2f'
          % r2_score(y_test, y_pred))
    np.array(y_test)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.2)
    plt.show()


def main():
    raw_data = get_data()
    process_data(raw_data)


if __name__ == "__main__":
    main()