#EXC 1
# ImageNet Preprocessing and Visualization using a Pre-trained CNN (ResNet50)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# ---------------- Task 1 & 2 : Load ImageNet subset (10 classes) ----------------
# Dataset is kept locally in the project directory:
#   datasets/tiny-imagenet-200/train/<synset>/images/*.JPEG   (ImageNet-200 subset)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "datasets", "tiny-imagenet-200", "train")

if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(f"Dataset not found at:\n{dataset_dir}")

# 10 selected ImageNet classes (synset id -> readable class name)
SELECTED_CLASSES = {
    "n01443537": "goldfish",
    "n02056570": "king penguin",
    "n02099601": "golden retriever",
    "n02106662": "german shepherd",
    "n02123394": "persian cat",
    "n02129165": "lion",
    "n02165456": "ladybug",
    "n02364673": "guinea pig",
    "n02403003": "ox",
    "n02480495": "orangutan",
}

IMAGES_PER_CLASS = 5      # small subset -> easy & fast run  (10 classes x 5 = 50 images)
IMG_SIZE = (224, 224)     # Task 4 : input size expected by ResNet50

# Collect images + original ImageNet class labels
images, labels = [], []
for synset, class_name in SELECTED_CLASSES.items():
    class_dir = os.path.join(dataset_dir, synset, "images")
    if not os.path.isdir(class_dir):
        print("Missing class folder, skipped:", synset)
        continue
    for fname in sorted(os.listdir(class_dir))[:IMAGES_PER_CLASS]:
        img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
        img = img.resize(IMG_SIZE)                       # Task 4 : Resize
        images.append(np.array(img))
        labels.append(class_name)                        # original label

images = np.array(images, dtype="float32")
print("Number of classes:", len(SELECTED_CLASSES))
print("Total images:", images.shape)

# ---------------- Task 3 : Train / Test split ----------------
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    images, labels, test_size=0.25, random_state=42
)
print("Train set:", x_train_raw.shape[0], " Test set:", x_test_raw.shape[0])

# ---------------- Task 5 : Normalize pixel values for ResNet50 ----------------
x_train = preprocess_input(x_train_raw.copy())   # converts 0-255 RGB -> BGR, mean subtraction
x_test = preprocess_input(x_test_raw.copy())

# ---------------- Task 6 : Display random images with original labels ----------------
idx = np.random.choice(len(x_test_raw), 6, replace=False)
plt.figure(figsize=(12, 6))
for i, j in enumerate(idx):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test_raw[j].astype("uint8"))
    plt.title(y_test[j], fontsize=9)
    plt.axis("off")
plt.suptitle("Random ImageNet Images with Original Class Labels")
plt.show()

# ---------------- Task 7 : Load pre-trained CNN ----------------
model = ResNet50(weights="imagenet")
print("Model loaded:", model.name, "| total layers:", len(model.layers))

# ---------------- Task 8 & 9 : Pass images, get top-5 predictions ----------------
pred = model.predict(x_test)
decoded = decode_predictions(pred, top=5)

# ---------------- Task 10 & 11 : Predicted classes, probabilities, compare ----------------
for i in range(len(x_test)):
    print("\nTest Image", i + 1, "| Original label:", y_test[i])
    top1 = decoded[i][0][1]
    match = "CORRECT" if (top1.replace("_", " ") == y_test[i]) else "WRONG"
    print("Top-1 prediction:", top1, "->", match)
    for _, name, prob in decoded[i]:
        print(f"   {name:<25} {prob * 100:6.2f}%")

# Confidence bar chart for the first image
plt.figure(figsize=(8, 4))
names = [n for _, n, _ in decoded[0]][::-1]
probs = [p * 100 for _, _, p in decoded[0]][::-1]
plt.barh(names, probs, color="skyblue")
plt.xlabel("Confidence (%)")
plt.title(f"Top-5 Predictions | Original label: {y_test[0]}")
plt.tight_layout()
plt.show()

# ---------------- Task 12 : Preprocessing steps and output dimensions ----------------
print("\n===== Preprocessing Steps =====")
print("1. Loaded images from ImageNet subset (tiny-imagenet-200, 10 selected classes)")
print("2. Resized every image to 224 x 224 x 3")
print("3. Pixel range before normalization :",
      int(x_test_raw.min()), "-", int(x_test_raw.max()))
print("4. preprocess_input() : RGB->BGR + mean subtraction, range now",
      round(float(x_test.min()), 1), "to", round(float(x_test.max()), 1))

# Track dimensions at different stages of the CNN
stage_layers = ["conv1_conv", "conv3_block4_out", "conv5_block3_out", "avg_pool"]
stage_model = tf.keras.Model(
    model.input, [model.get_layer(l).output for l in stage_layers]
)
stages = stage_model.predict(x_test[:1])

print("\n===== Output Dimensions (for 1 image) =====")
print(f"{'Stage':<22} {'Shape'}")
print(f"{'Preprocessed input':<22} {x_test[:1].shape}")
for name, out in zip(stage_layers, stages):
    print(f"{name:<22} {out.shape}")
print(f"{'Final predictions':<22} {pred[:1].shape}  (1000 ImageNet classes)")

# ---------------- Intermediate representation visualization ----------------
features = model.get_layer("conv3_block4_out").output
feat_model = tf.keras.Model(model.input, features)
feat = feat_model.predict(x_test[:1])

plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(feat[0, :, :, i], cmap="gray")
    plt.title(f"Filter {i}", fontsize=8)
    plt.axis("off")
plt.suptitle("Intermediate Representations (conv3_block4_out)")
plt.show()



#EXC 2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import get_file
from PIL import Image
import os

# Load model
model = ResNet50(weights="imagenet")

# Load image
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
file = get_file("image.jpg", url)
img = Image.open(file).resize((224,224))

x = preprocess_input(np.array(img))
x = np.expand_dims(x, 0)

# Select 3 layers
layers = ["conv1_conv", "conv2_block3_out", "conv4_block6_out"]

# Create model for feature maps
outputs = [model.get_layer(l).output for l in layers]
feature_model = tf.keras.Model(model.input, outputs)

features = feature_model.predict(x)

# Display feature maps
for layer, feature in zip(layers, features):
    print(layer, feature.shape)

    plt.figure(figsize=(10,4))
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(feature[0,:,:,i], cmap="gray")
        plt.axis("off")
    plt.suptitle(layer)
    plt.show()

# Visualize first-layer filters
filters = model.get_layer("conv1_conv").get_weights()[0]

plt.figure(figsize=(10,4))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(filters[:,:,0,i], cmap="gray")
    plt.axis("off")
plt.suptitle("Convolution Filters")
plt.show()




#EXC 3

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import os

# 1. Load pretrained CNN without classifier
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 2. Images and labels
urls = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
]

labels = [0, 1, 0, 1]
X = []

# 3. Feature extraction
for url in urls:
    file = get_file(os.path.basename(url), url)
    img = Image.open(file).resize((224,224))
    img = preprocess_input(np.array(img))
    X.append(img)

X = np.array(X)
X = model.predict(X)

print("Feature vector:", X.shape)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, random_state=42
)

# 5. Classification layer
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Prediction
y_pred = clf.predict(X_test)

# 7. Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

# 8. Confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred)
).plot()
plt.show()




#EXC 4

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import get_file
from PIL import Image

# 1. CNN encoder
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 2. Images and captions
urls = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
]

captions = [
    "a photo of a sunflower",
    "a photo of a person"
]

# 3. Extract CNN features
X = []

for url in urls:
    file = get_file(url.split("/")[-1], url)
    img = Image.open(file).resize((224,224))
    img = preprocess_input(np.array(img))
    X.append(img)

X = cnn.predict(np.array(X))

# 4. Add START and END tokens
captions = ["<start> " + c + " <end>" for c in captions]

# 5. Tokenize
tok = Tokenizer()
tok.fit_on_texts(captions)

seq = tok.texts_to_sequences(captions)
seq = pad_sequences(seq, padding="post")

vocab = len(tok.word_index) + 1

# 6. Input and output sequences
decoder_in = seq[:, :-1]
decoder_out = seq[:, 1:]

# 7. LSTM decoder
image = Input(shape=(2048,))
text = Input(shape=(None,))

x = Dense(128, activation="relu")(image)
x = tf.keras.layers.RepeatVector(decoder_in.shape[1])(x)
x = LSTM(128, return_sequences=True)(x)
output = Dense(vocab, activation="softmax")(x)

model = Model([image, text], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 8. Train
model.fit(
    [X, decoder_in],
    np.expand_dims(decoder_out, -1),
    epochs=10
)

print("Model trained")
print("Vocabulary:", tok.word_index)






#EXC 5

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from tensorflow.keras.models import Model

# 1. Text descriptions
texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a tiger"
]

# 2. Simple text representation
words = ["dog", "cat", "tiger"]
text_vectors = np.eye(3)

# 3. Generator
noise = Input(shape=(100,))
text = Input(shape=(3,))

x = Concatenate()([noise, text])
x = Dense(128, activation="relu")(x)
x = Dense(28*28, activation="sigmoid")(x)
output = Reshape((28,28,1))(x)

G = Model([noise, text], output)

# 4. Discriminator
image = Input(shape=(28,28,1))
text = Input(shape=(3,))

x = Flatten()(image)
x = Concatenate()([x, text])
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

D = Model([image, text], output)

D.compile(optimizer="adam", loss="binary_crossentropy")

# 5. Combined GAN
D.trainable = False
noise = Input(shape=(100,))
text = Input(shape=(3,))

fake = G([noise, text])
valid = D([fake, text])

GAN = Model([noise, text], valid)
GAN.compile(optimizer="adam", loss="binary_crossentropy")

# 6. Train
for epoch in range(100):
    noise = np.random.randn(3,100)
    fake = G.predict([noise, text_vectors], verbose=0)

    real = np.random.rand(3,28,28,1)

    D.train_on_batch([real,text_vectors], np.ones((3,1)))
    D.train_on_batch([fake,text_vectors], np.zeros((3,1)))

    GAN.train_on_batch(
        [np.random.randn(3,100), text_vectors],
        np.ones((3,1))
    )

# 7. Generate image
noise = np.random.randn(1,100)
text = np.array([[1,0,0]])       # dog

img = G.predict([noise,text], verbose=0)

plt.imshow(img[0,:,:,0], cmap="gray")
plt.title("Generated Image: Dog")
plt.axis("off")
plt.show()