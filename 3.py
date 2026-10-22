
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# 1. Load dataset
path = Path(r"D:\ImageNet")
folders = sorted([f for f in path.iterdir() if f.is_dir()])[:10]
X, labels = [], []

for folder in folders:
    imgs = [f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    for file in imgs[:100]:
        img = image.load_img(file, target_size=(224, 224))
        arr = image.img_to_array(img)
        X.append(preprocess_input(np.expand_dims(arr, axis=0))[0])
        labels.append(folder.name)

X = np.array(X, dtype=np.float32)

# 2. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y_cat = to_categorical(y)

print("Classes:", encoder.classes_)
print("Images:", len(X))

# 3. Extract ResNet50 features
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
features = base.predict(X, batch_size=16, verbose=1)

print("Feature shape:", features.shape)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)
y_train = to_categorical(y_train, num_classes=len(encoder.classes_))

# 5. Train classifier
model = Sequential([
    Dense(128, activation="relu", input_shape=(features.shape[1],)),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=16)

# 6. Evaluate model
pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_test, pred, average="macro", zero_division=0))
print("F1-score:", f1_score(y_test, pred, average="macro", zero_division=0))

# 7. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=encoder.classes_,
    xticks_rotation=90, cmap="Blues"
)
plt.tight_layout()
plt.show()
