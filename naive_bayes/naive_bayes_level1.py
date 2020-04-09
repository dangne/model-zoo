from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_digits



def main():
    # Load iris dataset for classification
    X, y = load_digits(return_X_y=True)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Declare Logistic Regression model
    model = GaussianNB()
    
    # Fit model to train set
    model.fit(x_train, y_train)
    
    # Evalute performance on test set
    pred = model.predict(x_test)
    print('Accuracy = ', accuracy_score(pred, y_test))

    return 'Done'

if __name__ == '__main__':
    main()
