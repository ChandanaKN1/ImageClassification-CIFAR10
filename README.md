# mnist-classifier


##  Model Architecture

- Input layer: 784 neurons (flattened 28x28 image)
- Dense output layer: 10 neurons with `sigmoid` activation

##  Libraries Used

- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

##  How It Works

1. Load and normalize MNIST dataset.
2. Flatten the image data (28x28 → 784).
3. Train a neural network with one dense layer.
4. Evaluate accuracy and visualize results using a confusion matrix.

##  Results

After 5 epochs of training, the model achieves around **91–92% accuracy** on the test set. A confusion matrix is also plotted to analyze classification performance.

##  Sample Code Snippet

```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train_flattened, y_train, epochs=5)
