
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

# 1. Load dataset
path = Path(r"D:\ImageNet")
folders = sorted([f for f in path.iterdir() if f.is_dir()])[:10]

train_images, test_images = [], []

for folder in folders:
    imgs = [f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.seed(42)
    random.shuffle(imgs)
    imgs = imgs[:100]

    if len(imgs) >= 2:
        train, test = train_test_split(
            imgs, test_size=0.2, random_state=42
        )
        train_images.extend(train)
        test_images.extend(test)

print("Training images:", len(train_images))
print("Testing images:", len(test_images))

# 2. Load pretrained ResNet50
model = ResNet50(weights="imagenet")
feature_model = Model(
    model.input,
    model.get_layer("conv3_block4_out").output
)

# 3. Test 5 random images
for file in random.sample(test_images, min(5, len(test_images))):
    img = image.load_img(file, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, axis=0))

    # 4. Predict top-5 classes
    predictions = model.predict(arr, verbose=0)
    top5 = decode_predictions(predictions, top=5)[0]

    print("\nActual class:", file.parent.name)
    print("Input shape:", arr.shape)
    print("Top-5 predictions:")

    for _, label, score in top5:
        print(label, ":", round(float(score) * 100, 2), "%")

    # 5. Display original image
    plt.imshow(img)
    plt.title("Actual: " + file.parent.name +
              "\nPredicted: " + top5[0][1])
    plt.axis("off")
    plt.show()

    # 6. Display intermediate feature maps
    features = feature_model.predict(arr, verbose=0)
    print("Feature map shape:", features.shape)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(features[0, :, :, i], cmap="viridis")
        ax.set_title(f"Feature {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
