import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import correlate2d


imag = plt.imread("bun.jpg")

fig, ax = plt.subplots(3,4, figsize=(10,10))
greyscale = imag.mean(axis=2)


s1 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0 ,1],
])
s2 = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0],
])
s3 = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
])
s4 = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2],
])

conv_1 = convolve(greyscale, s1)
conv_2 = convolve(greyscale, s2)
conv_3 = convolve(greyscale, s3)
conv_4 = convolve(greyscale, s4)


def corr(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    padded_image =  np.pad(image, pad_width=1, mode="edge") 
    res = np.zeros(image.shape)
    for row_idx, row in enumerate(image):
        for col_idx, _ in enumerate(row):
            # res[row_idx, col_idx] = (image[row_idx,col_idx]*mask[0,0] 
            # + padded_image[row_idx, col_idx+1]*mask[0,1]
            # + padded_image[row_idx, col_idx+2]*mask[0,2]
            # + padded_image[row_idx+1,col_idx]*mask[1,0]
            # + padded_image[row_idx+1, col_idx+1]*mask[1,1]
            # + padded_image[row_idx+1, col_idx+2]*mask[1,2]
            # + padded_image[row_idx+2,col_idx]*mask[2,0]
            # + padded_image[row_idx, col_idx+1]*mask[2,1]
            # + padded_image[row_idx+2, col_idx+2]*mask[2,2]
            # 
            # )
            import ipdb; ipdb.set_trace()
            res[row_idx, col_idx] = np.sum(padded_image[row_idx:row_idx+3, col_idx:col_idx+3] * mask)
        return res

import ipdb; ipdb.set_trace()
x = corr(greyscale, s1)


ax[0,0].imshow(conv_1, cmap="gray")
ax[0,1].imshow(conv_2, cmap="gray")
ax[0,2].imshow(conv_3, cmap="gray")
ax[0,3].imshow(conv_4, cmap="gray")

ax[0,0].set_title("S1")
ax[0,1].set_title("S2")
ax[0,2].set_title("S3")
ax[0,3].set_title("S4")

fig.tight_layout()
fig.savefig("result.png")