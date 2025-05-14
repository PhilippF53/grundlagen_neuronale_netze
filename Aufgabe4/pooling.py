import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("TkAgg")

# load image
img = Image.open("./Aufgabe4/data/sample-pgm.pgm")
img_array = np.array(img)
img_height = img_array.shape[0]
img_width = img_array.shape[1]
pooled_img_height = img_height // 2
pooled_img_width = img_width // 2
stride = 2
pooled_img_array = np.zeros((pooled_img_height, pooled_img_width))

# pool image
for i in range(0, img_width - 1, stride):
    for j in range(0, img_height - 1, stride):
        pooled_img_array[j // stride, i // stride] = np.mean(
            img_array[j : j + stride, i : i + stride]
        )

# plot images
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

axes[0].imshow(img_array, cmap="gray")
axes[0].set_xlabel("Original Image")

axes[1].imshow(pooled_img_array, cmap="gray")
axes[1].set_xlabel("Pooled Image")

plt.show()
