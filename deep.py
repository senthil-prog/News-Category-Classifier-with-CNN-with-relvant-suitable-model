import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# ======================================================
# 1. LOAD DATASET (AG News-like synthetic dataset)
# ======================================================
data = {
    "text": [
        "Stock markets rise amid economic optimism",
        "Local team wins championship after thrilling match",
        "New smartphone release features advanced AI camera",
        "Government announces new education policy reforms",
        "Tech stocks fall due to global supply chain issues",
        "Football league schedules announced for next season",
        "Breakthrough in quantum computing research",
        "Election results show major political shift",
        "Startup launches innovative electric vehicle",
        "Tennis star wins grand slam title"
    ] * 100,  # replicate to increase size
    "category": [
        "Business", "Sports", "Technology", "Politics",
        "Business", "Sports", "Technology", "Politics",
        "Technology", "Sports"
    ] * 100
}

df = pd.DataFrame(data)

# ======================================================
# 2. LABEL ENCODING
# ======================================================
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["category"])

num_classes = len(label_encoder.classes_)

# ======================================================
# 3. TEXT TOKENIZATION & PADDING
# ======================================================
max_words = 10000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=max_len)

y = df["label"].values

# ======================================================
# 4. TRAIN-TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# 5. CNN MODEL (TextCNN)
# ======================================================
input_layer = Input(shape=(max_len,))

embedding = Embedding(
    input_dim=max_words,
    output_dim=128,
    input_length=max_len
)(input_layer)

# Multi-kernel CNN
conv_3 = Conv1D(filters=128, kernel_size=3, activation="relu")(embedding)
conv_4 = Conv1D(filters=128, kernel_size=4, activation="relu")(embedding)
conv_5 = Conv1D(filters=128, kernel_size=5, activation="relu")(embedding)

pool_3 = GlobalMaxPooling1D()(conv_3)
pool_4 = GlobalMaxPooling1D()(conv_4)
pool_5 = GlobalMaxPooling1D()(conv_5)

concat = Concatenate()([pool_3, pool_4, pool_5])

dropout = Dropout(0.5)(concat)
dense = Dense(128, activation="relu")(dropout)
output = Dense(num_classes, activation="softmax")(dense)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================================
# 6. TRAINING
# ======================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# ======================================================
# 7. EVALUATION
# ======================================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=label_encoder.classes_
))

# ======================================================
# 8. CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# ======================================================
# 9. TRAINING CURVES
# ======================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.show()
