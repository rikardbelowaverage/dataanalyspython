import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd



def linear_prog_diabetes():
    diabetes = datasets.load_diabetes()  # Load dataset
    # print(diabetes.DESCR)
    X = diabetes.data  # X for data
    Y = diabetes.target  # Y for target

    # Alternative solution
    # X , Y = datasets.load_diabetes(return_X_y=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2)  # 80% goes to training set, 20% goes to test set

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Coeffs:", model.coef_) #Coeeffs are related to features printed below
    print(diabetes.feature_names) #Here are features that are used as inputs
    print("Intercept:", model.intercept_)
    print("Mean square error : %.2f"
          % mean_squared_error(Y_test, Y_pred))
    print("Coefficient of determination (R^2): %.2f"
          % r2_score(Y_test, Y_pred))

    np.array(Y_test)
    sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.5)
    plt.show()


def linear_prog_boston():
    BostonHousing = pd.read_csv("BostonHousing.csv") #panda read csv type, var = BostonHousing
    Y = BostonHousing.medv #name of last column
    X = BostonHousing.drop(['medv'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print('Coefficients:', model.coef_)
    print(BostonHousing.head())
    print('Intercept:', model.intercept_)
    print('Mean squared error (MSE): %.2f'
          % mean_squared_error(Y_test, Y_pred))
    print('Coefficient of determination (R^2): %.2f'
          % r2_score(Y_test, Y_pred))
    #medv(Y) = coef[0]*crim + coef[1]*zn.... + Y intercept
    # Y = k*x + m
    np.array(Y_test)
    sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.2)
    plt.show()

def main():
    linear_prog_boston()
    linear_prog_diabetes()

if __name__ == "__main__":
    main()
