import forestpt

# Train the model
forestpt.train_model(num_epochs=10)

# Evaluate the model
forestpt.evaluate_model()

# Print the model summary
forestpt.model_summary()

# Save the model
forestpt.save_model('forest_model1.pth')

# Load the model
forestpt.load_model('forest_model1.pth')

# Predict using the model (example)
# image_bytes = ...  # Load or get the image bytes
# prediction = forestpt.predict(image_bytes)
# print(f"Prediction: {prediction}")
