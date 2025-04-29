#
#
#
# Your code to perform the above task.
# You can split your code into multiple cells.
#
#
#
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

# Load MNIST dataset and preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define and train a simple model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(28*28,)),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Adversarial generation functions
loss_object = tf.keras.losses.CategoricalCrossentropy()

def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * np.sign(gradient)
    adversarial_image = image + perturbation
    adversarial_image = np.clip(adversarial_image, 0, 1)
    return adversarial_image, perturbation

def create_adversarial_examples(model, images, labels, epsilon=0.1):
    images_tensor = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        tape.watch(images_tensor)
        predictions = model(images_tensor)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, images_tensor)
    adversarial_examples, perturbations = fgsm_attack(images, epsilon, gradients.numpy())
    
    return adversarial_examples, perturbations

# --- Now the important correction ---

epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
accuracies = []

# Use more test samples for better statistics
num_samples = 1000  
x_sample = x_test[:num_samples]
y_sample = y_test[:num_samples]

# Generate adversarial examples and evaluate for each epsilon
for epsilon in epsilon_values:
    print(f"\nEpsilon = {epsilon}")
    
    if epsilon == 0.0:
        # For epsilon 0.0, use original images (no perturbation)
        x_adv = x_sample
    else:
        x_adv, _ = create_adversarial_examples(model, x_sample, y_sample, epsilon)

    # Evaluate model
    loss_adv, accuracy_adv = model.evaluate(x_adv, y_sample, verbose=0)
    accuracies.append(accuracy_adv)
    print(f"Accuracy on adversarial examples: {accuracy_adv:.4f}")

# --- Plotting epsilon vs accuracy ---
plt.figure(figsize=(8,6))
plt.plot(epsilon_values, accuracies, marker='o', linestyle='--', color='blue')
plt.title('Model Accuracy vs Epsilon')
plt.xlabel('Epsilon (Perturbation Strength)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# --- Import libraries ---
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Load and preprocess the MNIST dataset ---
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Save clean labels for original training
y_train_clean = y_train.copy()

# --- Define a function to build the model ---
def build_model():
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(28*28,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Train on the original (clean) dataset ---
# One-hot encode labels
y_train_clean_onehot = keras.utils.to_categorical(y_train_clean, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

# Build and train the model
model = build_model()
original_history = model.fit(x_train, y_train_clean_onehot, epochs=10, batch_size=128, validation_data=(x_test, y_test_onehot))

# Evaluate the model before poisoning
loss_before, accuracy_before = model.evaluate(x_test, y_test_onehot)
print(f"\nAccuracy before data poisoning: {accuracy_before:.4f}")

# --- Poison the training data ---
# Set all training labels to 1
y_train_poisoned = np.ones_like(y_train_clean)

# One-hot encode poisoned labels
y_train_poisoned_onehot = keras.utils.to_categorical(y_train_poisoned, 10)

# --- Train on the poisoned dataset ---
# Build a new model (reset weights)
poisoned_model = build_model()

# Train on poisoned data
poisoned_history = poisoned_model.fit(x_train, y_train_poisoned_onehot, epochs=10, batch_size=128, validation_data=(x_test, y_test_onehot))

# Evaluate the model after poisoning
loss_after, accuracy_after = poisoned_model.evaluate(x_test, y_test_onehot)
print(f"\nAccuracy after data poisoning: {accuracy_after:.4f}")

# --- Compare accuracy before and after poisoning ---
print("\n=== Accuracy Comparison ===")
print(f"Accuracy before data poisoning: {accuracy_before:.4f}")
print(f"Accuracy after data poisoning: {accuracy_after:.4f}")

# --- (Optional) Plot Training and Validation Accuracy ---
plt.figure(figsize=(12,5))

# Plot for original training
plt.subplot(1,2,1)
plt.plot(original_history.history['accuracy'], label='Train Accuracy')
plt.plot(original_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Before Data Poisoning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot for poisoned training
plt.subplot(1,2,2)
plt.plot(poisoned_history.history['accuracy'], label='Train Accuracy')
plt.plot(poisoned_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model After Data Poisoning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
