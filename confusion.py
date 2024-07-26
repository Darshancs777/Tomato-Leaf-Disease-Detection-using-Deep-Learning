import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load the trained model
model = load_model("model.h5")

# Define the batch size for the test set
batch_size_test = 16

# Create an instance of ImageDataGenerator for the test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Verify the directory path
test_dir = 'D:\\Tomato Leaf Disease Detection using Deep Learning\\Dataset\\test'
if not os.path.exists(test_dir):
    raise ValueError(f"Test directory {test_dir} does not exist")

# Print the contents of the test directory
print("Contents of test directory:", os.listdir(test_dir))

# Print the contents of each subdirectory
for subdir in os.listdir(test_dir):
    subdir_path = os.path.join(test_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"Contents of {subdir_path}:", os.listdir(subdir_path))
    else:
        print(f"{subdir_path} is not a directory")

# Prepare the test set
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(128, 128),
                                            batch_size=batch_size_test,
                                            class_mode='categorical',
                                            shuffle=False)  # Important to set shuffle to False

# Check if the test set contains any images
if test_set.samples == 0:
    raise ValueError(f"No images found in the test directory {test_dir}")

# Predict using the trained model on the test set
y_pred_prob = model.predict(test_set, steps=int(np.ceil(test_set.samples / test_set.batch_size)))
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

# True labels
y_true = test_set.classes

# Compute confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_mat)

# Classification report
class_report = classification_report(y_true, y_pred, target_names=list(test_set.class_indices.keys()))
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=list(test_set.class_indices.keys()), yticklabels=list(test_set.class_indices.keys()))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
