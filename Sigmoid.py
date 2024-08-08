import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import gc

# Set environment variables to ensure TensorFlow uses GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)

# Paths to directories
fire_dir = ''
no_fire_dir = ''

# Function to load a subset of CSV files
def load_data(directory, max_files=50):
    data_list = []
    filenames = [f for f in os.listdir(directory) if f.endswith('.csv')]
    selected_files = random.sample(filenames, min(max_files, len(filenames)))

    for filename in selected_files:
        filepath = os.path.join(directory, filename)
        data = pd.read_csv(filepath)
        data = data.drop(columns=['row', 'col'])  # Drop columns 'row' and 'col'
        data = data.astype('float32')  # Convert data to float32
        data_list.append(data)

    return pd.concat(data_list, ignore_index=True)

fire_data = load_data(fire_dir, max_files=5)
no_fire_data = load_data(no_fire_dir, max_files=5)

# Combine both datasets
data = pd.concat([fire_data, no_fire_data], ignore_index=True)

# Separate features and labels
X = data.drop(columns=['Probability'])  # Adjust the column name based on the CSV data
y = data['Probability']

# Normalize or standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Garbage collection to free up memory
gc.collect()

# Define the custom sigmoid function
import tensorflow.keras.backend as K

def custom_sigmoid(x, a=0.2, b=0.0):
    return 1 / (1 + K.exp(-a * (x - b)))

# Define the model
model = Sequential()
model.add(tf.keras.Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))  # Output layer with a single neuron
model.add(Activation(lambda x: custom_sigmoid(x, a=0.5, b=0.0)))  # Use the custom sigmoid function here

# Set the learning rate
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('sigmoid.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Train the model with callbacks
num_epochs = 200
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=64, validation_split=0.25,
                    callbacks=[checkpoint, early_stopping])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Load the best model
new_model = tf.keras.models.load_model('best_model.keras', custom_objects={'<lambda>': lambda x: custom_sigmoid(x, a=0.5, b=0.0)})

# Predict on test data using the loaded model
y_pred = new_model.predict(X_test)

# Function to reshape predictions back to image dimensions
def create_heatmap(predictions, image_shape=(224, 224)):
    return predictions.reshape(image_shape)

# Assuming test data comes from a single image for simplicity
heatmap = create_heatmap(y_pred[:224 * 224])

# Visualize the heatmap
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Fire Probability Heatmap')
plt.show()

