import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Set vocabulary size
vocab_size = 10000  # Consider top 10,000 words
max_len = 200  # Max length of reviews (only 200 words of every review considered)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

#the x_train,x_test contains reviews in form of numbers that is word indexes
#the x_test , y_test contains labels that is 1 for positive review and 0 for negative

# Pad sequences to bring all reviews to same length
x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

"""embedding layer - converts word indexed into dense vectors of size whatever size we pass (128 in this case),
more complex output_dim then more training time , will capture more complex meanings
LSMT Layer1 , LSMT Layer2
Dense output layer - outputs 1 value between 0 and 1 (probablity of being a positive review),
sigmoid function used so if greater than 0.5 then positive , negative otherwise"""

# Build RNN model using LSTM
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#5 epochs , 64 reviews considered at a time to updagte weights
# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Predict a single review
def predict_review(index):
    review = x_test[index].reshape(1, max_len)
    prediction = model.predict(review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Prediction: {sentiment} ({prediction:.4f})")

# Test on an example review
predict_review(0)

