import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist

def main():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # expand dimension for the 3rd channel
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    
    model = Sequential([
        Input(shape=(x_train[0].shape)),
        Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.summary()
    
    model.fit(x_train, y_train)
    
    model.evaluate(x_test, y_test)

    return 'Done'

if __name__ == '__main__':
    main()
