import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


imag = plt.imread("vessel.jpeg")

fig, ax = plt.subplots(3,4, figsize=(10,10))
greyscale = imag.mean(axis=2).astype(int)

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

fig.tight_layout()
ax[0,0].imshow(conv_1, cmap="gray")
ax[0,1].imshow(conv_2, cmap="gray")
ax[0,2].imshow(conv_3, cmap="gray")
ax[0,3].imshow(conv_4, cmap="gray")


def correlate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    res = np.zeros([dim-2 for dim in image.shape], dtype=int)
    x,y = res.shape
    for row in range(x):
        for col in range(y):
            res[row,col] = np.sum(image[row:row+3, col:col+3] * mask)

    return res

ss1 = correlate(greyscale, s1)
ss2 = correlate(greyscale, s2)
ss3 = correlate(greyscale, s3)
ss4 = correlate(greyscale, s4)

ax[1,0].imshow(ss1, cmap="gray")
ax[1,2].imshow(ss2, cmap="gray")
ax[1,1].imshow(ss3, cmap="gray")
ax[1,3].imshow(ss4, cmap="gray")

def conv_x(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape[0] < mask.shape[0]:
        print("reverse")
        return conv_x(mask, image)

    mask_shape = mask.shape
    img_x, img_y = image.shape
    mask_x, mask_y = mask.shape
    img_x -= mask_x - 1
    img_y -= mask_y - 1
    res = np.zeros((img_x, img_y), dtype=int)

    m = np.flipud(mask)
    m = np.fliplr(m)
    x,y = res.shape
    try:
        for row in range(x):
            for col in range(y):
                res[row,col] = np.sum(image[row:row+mask_shape[0], col:col+mask_shape[1]] * m)
    except ValueError:
        import ipdb; ipdb.set_trace()
    print(res.shape)
    return res

s5 = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2],
    [0, -1, -2],
])
print(f"{greyscale.shape=}")
convs1 = conv_x(greyscale, s1)
ax[2,0].imshow(convs1, cmap="gray")
convs2 = conv_x(greyscale, s2)
ax[2,1].imshow(convs2, cmap="gray")
convs3 = conv_x(greyscale, s3)
ax[2,2].imshow(convs3, cmap="gray")
convs4 = conv_x(greyscale, s4)
convs4 = conv_x(greyscale, s5)
convs4 = conv_x(s4, greyscale)
ax[2,3].imshow(convs4, cmap="gray")

fig.savefig("result.png")


