
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load images and create class-based captions
path = Path(r"D:\ImageNet")
images, captions = [], []

for folder in sorted([p for p in path.iterdir() if p.is_dir()])[:5]:
    files = [
        file for file in folder.iterdir()
        if file.suffix.lower() in
        (".jpg", ".jpeg", ".png", ".bmp")
    ][:10]

    for file in files:
        images.append(file)
        captions.append(
            "<start> a photo of " +
            folder.name.replace("_", " ").lower() + " <end>"
        )

# 2. Extract image features using ResNet50
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
features = []

for file in images:
    img = image.load_img(file, target_size=(224, 224))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    features.append(cnn.predict(arr, verbose=0)[0])

features = np.array(features)

# 3. Tokenize captions
tokenizer = Tokenizer(filters="", lower=True)
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)
vocab = len(tokenizer.word_index) + 1
max_len = max(len(s) for s in sequences)

# 4. Prepare next-word training data
img_data, inputs, targets = [], [], []

for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        img_data.append(features[i])
        inputs.append(seq[:j])
        targets.append(seq[j])

inputs = pad_sequences(inputs, maxlen=max_len - 1, padding="post")
targets = tf.keras.utils.to_categorical(targets, num_classes=vocab)
img_data = np.array(img_data)

# 5. Build CNN-LSTM model
image_input = Input(shape=(2048,))
word_input = Input(shape=(max_len - 1,))

a = Dense(128, activation="relu")(image_input)
a = RepeatVector(max_len - 1)(a)
b = Embedding(vocab, 128, mask_zero=True)(word_input)

x = Add()([a, b])
x = LSTM(128)(x)
output = Dense(vocab, activation="softmax")(x)

model = Model([image_input, word_input], output)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 6. Train the model
model.fit([img_data, inputs], targets, epochs=10, batch_size=16)

# 7. Generate a caption
def generate_caption(file):
    img = image.load_img(file, target_size=(224, 224))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    feature = cnn.predict(arr, verbose=0)

    caption = "<start>"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_len - 1, padding="post")
        pred = model.predict([feature, seq], verbose=0)[0]
        word = tokenizer.index_word.get(np.argmax(pred), "")

        if not word or word == "<end>":
            break

        caption += " " + word

    return caption.replace("<start>", "").strip()

# 8. Display image and generated caption
file = images[0]
print("Expected caption:", captions[0].replace("<start> ", "").replace(" <end>", ""))
print("Generated caption:", generate_caption(file))

plt.imshow(image.load_img(file))
plt.title(generate_caption(file))
plt.axis("off")
plt.show()
