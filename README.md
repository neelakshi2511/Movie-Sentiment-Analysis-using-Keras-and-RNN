# Movie-Sentiment-Analysis-using-Keras-and-RNN

tried understanding RNN implementation on IMDB Movie dataset(imported from Keras)

This implementation performs **Sentiment Analysis** using an LSTM-based RNN on the **IMDb movie reviews dataset**.  
The model classifies reviews as **Positive** or **Negative** based on the words in the review.

We build a neural network that learns from sequences of word indices, uses word embeddings, and applies LSTM layers to capture complex patterns in the data for accurate binary classification.

1. **Load IMDb Dataset**
   - Used top 10,000 most frequent words for modeling.
   - Loaded the reviews (`x_train`, `x_test`) and corresponding labels (`y_train`, `y_test`).

2. **Preprocessing**
   - Reviews are already tokenized into word indices.
   - Padded all reviews to a fixed length (`max_len = 200`) to ensure uniform input size.

3. **Build the LSTM Model**
   - **Embedding Layer**: Maps word indices into dense vectors of 128 dimensions.
   - **First LSTM Layer**: 64 memory units, returns sequences for stacking.
   - **Second LSTM Layer**: 32 memory units, captures sequential dependencies further.
   - **Dense Output Layer**: Single neuron with `sigmoid` activation to output a probability between 0 and 1 (positive or negative review).

4. **Compile the Model**
   - Loss: `binary_crossentropy` (suitable for binary classification).
   - Optimizer: `adam`.
   - Metric: `accuracy`.

5. **Train the Model**
   - Trained for 5 epochs with a batch size of 64.
   - Used validation data (`x_test`, `y_test`) to monitor performance.

6. **Evaluate the Model**
   - Achieved test accuracy after training.
   - Printed evaluation score on unseen test data.

7. **Predict Single Review**
   - `predict_review(index)` function:
     - Takes an index from `x_test`.
     - Predicts whether the review is positive or negative based on model output.

This will print whether the selected review is **Positive** or **Negative** along with the prediction probability.

