"""
Basic RNN Tutorial using Keras
Covers: sequence prediction with SimpleRNN
"""

# Install tensorflow if not already installed (run this in Jupyter)
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "--quiet"])

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# ─────────────────────────────────────────────
# PART 1: Basic RNN — Predict Next Number
# ─────────────────────────────────────────────

print("=" * 50)
print("PART 1: Basic RNN")
print("=" * 50)

# Step 2: Create dummy sequential data
# Pattern: [n, n+1, n+2] → n+3
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
])

y = np.array([4, 5, 6, 7])

# Step 3: Reshape to (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"X shape: {X.shape}")  # (4, 3, 1)

# Step 4: Build the RNN model
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(3, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Step 5: Train
model.fit(X, y, epochs=200, verbose=0)
print("Training complete.")

# Step 6: Test prediction — expect ≈ 8
test = np.array([[5, 6, 7]]).reshape((1, 3, 1))
prediction = model.predict(test)
print(f"Input: [5, 6, 7] → Predicted: {prediction[0][0]:.4f}  (expected ≈ 8)")


# ─────────────────────────────────────────────
# PART 3: Modified Dataset (×10 pattern)
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("PART 3: Modified Dataset (multiples of 10)")
print("=" * 50)

X2 = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
])

y2 = np.array([40, 50, 60])

X2 = X2.reshape((X2.shape[0], X2.shape[1], 1))
print(f"X2 shape: {X2.shape}")

model2 = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(3, 1)),
    Dense(1)
])

model2.compile(optimizer='adam', loss='mse')
model2.fit(X2, y2, epochs=500, verbose=0)
print("Training complete.")

# Predict [40, 50, 60] → expect ≈ 70
test2 = np.array([[40, 50, 60]]).reshape((1, 3, 1))
prediction2 = model2.predict(test2)
print(f"Input: [40, 50, 60] → Predicted: {prediction2[0][0]:.4f}  (expected ≈ 70)")


# ─────────────────────────────────────────────
# PART 4 (Optional): Simple Character Prediction
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("PART 4: Simple Character Pattern Prediction")
print("=" * 50)

# Encode characters as integers
# Pattern: "abcd" → each 3-char window predicts the next char
text = "abcdabcdabcd"
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

vocab_size = len(chars)
seq_len = 3

# Build input/output pairs
sequences = []
targets = []
for i in range(len(text) - seq_len):
    sequences.append([char_to_idx[c] for c in text[i:i + seq_len]])
    targets.append(char_to_idx[text[i + seq_len]])

X3 = np.array(sequences, dtype=np.float32).reshape(-1, seq_len, 1) / vocab_size
y3 = np.array(targets)

# One-hot encode targets
from tensorflow.keras.utils import to_categorical
y3_cat = to_categorical(y3, num_classes=vocab_size)

model3 = Sequential([
    SimpleRNN(32, activation='relu', input_shape=(seq_len, 1)),
    Dense(vocab_size, activation='softmax')
])

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(X3, y3_cat, epochs=300, verbose=0)
print("Training complete.")

# Predict: "abc" → expect 'd'
test_seq = "abc"
test3 = np.array([[char_to_idx[c] for c in test_seq]], dtype=np.float32).reshape(1, seq_len, 1) / vocab_size
pred3 = model3.predict(test3)
predicted_char = idx_to_char[np.argmax(pred3)]
print(f"Input: '{test_seq}' → Predicted next char: '{predicted_char}'  (expected 'd')")
