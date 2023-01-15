import copy
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import (
    disk,
    binary_closing,
    binary_opening,
    binary_dilation,
    binary_erosion,
)


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    img = img - np.min(img)
    img /= np.max(img)
    return img


# fig, ax = plt.subplots(1, 4, figsize=(15, 15))
# image = plt.imread("fingerprint.jpg")
# image = image.mean(axis=2)
# normalized_image = normalize(image)
# # print(normalized_image[400:500, 400:450])
# ax[0].imshow(normalized_image, cmap="gray")
# threshold = copy.deepcopy(normalized_image)
# threshold[normalized_image < 0.6] = 1
# threshold[normalized_image >= 0.6] = 0
# print(threshold)
# print(normalized_image.shape)
# footprint = disk(1, float)
# erosion = binary_erosion(threshold, footprint)
# closing = binary_closing(threshold, footprint)
# dilation = binary_dilation(threshold, footprint)
# opening = binary_opening(threshold, footprint)
# ax[0].imshow(erosion, cmap="binary")
# ax[1].imshow(dilation, cmap="binary")
# ax[2].imshow(opening, cmap="binary")
# ax[3].imshow(closing, cmap="binary")

# fig.savefig("result.jpg")


# Zadanie 2
fig, ax = plt.subplots(4, 6, figsize=(15, 15))
image = plt.imread("fingerprint.jpg")
image = image.mean(axis=2)
normalized_image = normalize(image)
# print(normalized_image[400:500, 400:450])
threshold = copy.deepcopy(normalized_image)
threshold[normalized_image < 0.6] = 1
threshold[normalized_image >= 0.6] = 0
print(threshold)
print(normalized_image.shape)
footprint = disk(1, float)


def generate_images(image, footprint):
    erosion = binary_erosion(image, footprint)
    closing = binary_closing(image, footprint)
    dilation = binary_dilation(image, footprint)
    opening = binary_opening(image, footprint)

    return erosion, closing, dilation, opening


composition = np.zeros((900, 644, 3))
erosion, closing, dilation, opening = generate_images(threshold, footprint)
ax[0, 0].imshow(footprint, cmap="binary")
ax[0, 1].imshow(erosion, cmap="binary")
ax[0, 2].imshow(dilation, cmap="binary")
ax[0, 3].imshow(opening, cmap="binary")
ax[0, 4].imshow(closing, cmap="binary")
composition[:, :, 0] = dilation
composition[:, :, 1] = opening
composition[:, :, 2] = closing
ax[0, 5].imshow(composition)
footprint = disk(5)
erosion, closing, dilation, opening = generate_images(threshold, footprint)
ax[1, 0].imshow(footprint, cmap="binary")
ax[1, 1].imshow(erosion, cmap="binary")
ax[1, 2].imshow(dilation, cmap="binary")
ax[1, 3].imshow(opening, cmap="binary")
ax[1, 4].imshow(closing, cmap="binary")
composition[:, :, 0] = dilation
composition[:, :, 1] = opening
composition[:, :, 2] = closing
ax[1, 5].imshow(composition)
footprint = np.ones((10, 1), float)
erosion, closing, dilation, opening = generate_images(threshold, footprint)
ax[2, 0].imshow(footprint, cmap="binary_r")
ax[2, 1].imshow(erosion, cmap="binary")
ax[2, 2].imshow(dilation, cmap="binary")
ax[2, 3].imshow(opening, cmap="binary")
ax[2, 4].imshow(closing, cmap="binary")
composition[:, :, 0] = dilation
composition[:, :, 1] = opening
composition[:, :, 2] = closing
ax[2, 5].imshow(composition)
footprint = np.ones((1, 10), float)
erosion, closing, dilation, opening = generate_images(threshold, footprint)
ax[3, 0].imshow(footprint, cmap="binary_r")
ax[3, 1].imshow(erosion, cmap="binary")
ax[3, 2].imshow(dilation, cmap="binary")
ax[3, 3].imshow(opening, cmap="binary")
ax[3, 4].imshow(closing, cmap="binary")
composition[:, :, 0] = dilation
composition[:, :, 1] = opening
composition[:, :, 2] = closing
ax[3, 5].imshow(composition)
fig.tight_layout()
fig.savefig("result.jpg")

# Zadanie 3
