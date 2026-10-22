
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Concatenate
)
from tensorflow.keras.models import Model

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Text labels (one-hot encoded)
texts = ["dog", "cat", "tiger"]
text_vectors = np.eye(3, dtype=np.float32)

# 2. Generator
noise_input = Input(shape=(100,), name="noise")
gen_text_input = Input(shape=(3,), name="text")

x = Concatenate()([noise_input, gen_text_input])
x = Dense(128, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(28 * 28, activation="sigmoid")(x)

generated_image = Reshape((28, 28, 1))(x)

G = Model(
    [noise_input, gen_text_input],
    generated_image,
    name="Generator"
)

# 3. Discriminator
image_input = Input(shape=(28, 28, 1), name="image")
disc_text_input = Input(shape=(3,), name="text")

x = Flatten()(image_input)
x = Concatenate()([x, disc_text_input])
x = Dense(128, activation="relu")(x)
validity = Dense(1, activation="sigmoid")(x)

D = Model(
    [image_input, disc_text_input],
    validity,
    name="Discriminator"
)

D.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002),
    loss="binary_crossentropy"
)

# 4. Combined GAN
D.trainable = False

gan_noise_input = Input(shape=(100,), name="gan_noise")
gan_text_input = Input(shape=(3,), name="gan_text")

fake_image = G([gan_noise_input, gan_text_input])
fake_validity = D([fake_image, gan_text_input])

GAN = Model(
    [gan_noise_input, gan_text_input],
    fake_validity,
    name="Conditional_GAN"
)

GAN.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002),
    loss="binary_crossentropy"
)

# Re-enable discriminator for its own training
D.trainable = True

# 5. Training demonstration
# NOTE: Random images are placeholders, not real animal images.
batch_size = 3
epochs = 100

for epoch in range(epochs):

    # Random placeholder images in the range [0, 1]
    real_images = np.random.rand(
        batch_size, 28, 28, 1
    ).astype(np.float32)

    real_text = text_vectors.copy()

    # Generate fake images
    noise = np.random.normal(
        0, 1, (batch_size, 100)
    ).astype(np.float32)

    fake_images = G.predict(
        [noise, real_text], verbose=0
    )

    # Train discriminator
    real_labels = np.ones((batch_size, 1), dtype=np.float32)
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32)

    d_loss_real = D.train_on_batch(
        [real_images, real_text], real_labels
    )

    d_loss_fake = D.train_on_batch(
        [fake_images, real_text], fake_labels
    )

    # Train generator through the combined GAN
    noise = np.random.normal(
        0, 1, (batch_size, 100)
    ).astype(np.float32)

    g_loss = GAN.train_on_batch(
        [noise, real_text], real_labels
    )

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"D real loss: {float(np.mean(d_loss_real)):.4f} | "
            f"D fake loss: {float(np.mean(d_loss_fake)):.4f} | "
            f"G loss: {float(np.mean(g_loss)):.4f}"
        )

# 6. Generate an image conditioned on "dog"
noise = np.random.normal(
    0, 1, (1, 100)
).astype(np.float32)

dog_text = np.array([[1, 0, 0]], dtype=np.float32)

img = G.predict([noise, dog_text], verbose=0)

plt.imshow(img[0, :, :, 0], cmap="gray", vmin=0, vmax=1)
plt.title("Generated Image: Dog (Untrained Demo)")
plt.axis("off")
plt.show()
