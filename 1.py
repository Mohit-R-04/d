
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import (
    ResNet50, preprocess_input, decode_predictions
)
from tensorflow.keras.models import Model

# 1. Load ImageNet subset
path = Path(r"D:\ImageNet")
folders = sorted([f for f in path.iterdir() if f.is_dir()])[:10]
train_images, test_images = [], []

random.seed(42)

for folder in folders:
    imgs = [f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    random.shuffle(imgs)
    imgs = imgs[:100]

    if len(imgs) >= 2:
        train, test = train_test_split(
            imgs, test_size=0.2, random_state=42
        )
        train_images.extend(train)
        test_images.extend(test)

print("Classes:", [f.name for f in folders])
print("Training images:", len(train_images))
print("Testing images:", len(test_images))

# 2. Load pretrained ResNet50
model = ResNet50(weights="imagenet")
feature_model = Model(
    model.input, model.get_layer("conv3_block4_out").output
)

# 3. Test five random images
for file in random.sample(test_images, min(5, len(test_images))):
    img = image.load_img(file, target_size=(224, 224))
    arr = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(arr, axis=0))

    # 4. Predict top-5 ImageNet classes
    predictions = model.predict(x, verbose=0)
    top5 = decode_predictions(predictions, top=5)[0]

    print("\nActual class:", file.parent.name)
    print("Original image shape:", arr.shape)
    print("Preprocessed image shape:", x.shape)
    print("Prediction output shape:", predictions.shape)
    print("Top-5 predictions:")

    for _, label, score in top5:
        print(f"{label}: {score * 100:.2f}%")

    print("Top-1 prediction:", top5[0][1])
    print("Label comparison:",
          "MATCH" if file.parent.name.lower() in top5[0][1].lower()
          else "NO MATCH")

    # 5. Display original image
    plt.imshow(img)
    plt.title(f"Actual: {file.parent.name}\nPredicted: {top5[0][1]}")
    plt.axis("off")
    plt.show()

    # 6. Display intermediate feature maps
    features = feature_model.predict(x, verbose=0)
    print("Feature map shape:", features.shape)

    plt.figure(figsize=(10, 5))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(features[0, :, :, i], cmap="viridis")
        plt.title(f"Feature {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
