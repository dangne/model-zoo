import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
     
    # Reshape input
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')
     
    # Normalize input
    x_train /= 255
    x_test /= 255
     
    # Define model using Sequential API
    model = Sequential([
        Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
     
    # Compile model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
     
    model.summary()
     
    # Fit model to the training data
    model.fit(x_train, y_train)
     
    # Perform prediction
    model.evaluate(x_test, y_test)

    return 'Done'

if __name__ == '__main__':
    main()
