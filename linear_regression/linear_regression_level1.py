from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.datasets import make_regression 



def main():

    # Create dummy dataset for linear regression
    X, y = make_regression(n_features=1, n_informative=1, noise=10)
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Declare Linear Regression model
    model = LinearRegression()
    
    # Fit model to train set
    model.fit(x_train, y_train)
    
    # Evalute performance on test set
    pred = model.predict(x_test)
    print('Mean squared error = ', mean_squared_error(pred, y_test))

    return 'Done'



if __name__ == '__main__':
    main()
