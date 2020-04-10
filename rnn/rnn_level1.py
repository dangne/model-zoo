import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences



def main():
    VOCAB_SIZE = 5000

    # Load IMDB dataset for binary sentiment classification
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = VOCAB_SIZE)

    # Padding and truncating sequences
    x_train = pad_sequences(x_train, maxlen=500)
    x_test = pad_sequences(x_test, maxlen=500)

    # Define model
    model = Sequential([
        Embedding(VOCAB_SIZE, 64),
        SimpleRNN(32, return_sequences=True),
        SimpleRNN(32),
        Dense(1, activation='sigmoid')
    ])
    
    # Print model summary 
    model.summary()
    
    # Compile model
    model.compile(optimizer='adam',
	            loss='binary_crossentropy',
	            metrics=['accuracy'])
    
    # Fit model to train set
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=3)
    
    # Evalute model's performance
    model.evaluate(x_test, y_test)
    
    return 'Done'



if __name__ == '__main__':
    main()
