from matplotlib import pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon
from skimage.transform import iradon



fig, ax = plt.subplots(4, 4, figsize=(10, 10))
img = shepp_logan_phantom()


def foo(n: int, image: np.ndarray, ):
    lin = np.linspace(0, 180, n)
    radon_ = radon(img, lin)
    iradon_ = iradon(radon_, theta=lin)
    err = img - iradon_
    return image, radon_, iradon_, err

img1, radon1, iradon1, err1 = foo(10, img)
img2, radon2, iradon2, err2 = foo(30, img)
img3, radon3, iradon3, err3 = foo(100, img)
img4, radon4, iradon4, err4 = foo(1000, img)

ax[0,0].imshow(img1, cmap="binary_r")
ax[0,1].imshow(radon1, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[0,2].imshow(iradon1, cmap="binary_r")
ax[0,3].imshow(err1, cmap="binary")
ax[1,0].imshow(img2, cmap="binary_r")
ax[1,1].imshow(radon2, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[1,2].imshow(iradon2, cmap="binary_r")
ax[1,3].imshow(err3, cmap="binary")
ax[2,0].imshow(img3, cmap="binary_r")
ax[2,1].imshow(radon3, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[2,2].imshow(iradon3, cmap="binary_r")
ax[2,3].imshow(err3, cmap="binary")
ax[3,0].imshow(img4, cmap="binary_r")
ax[3,1].imshow(radon4, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[3,2].imshow(iradon4, cmap="binary_r")
ax[3,3].imshow(err4, cmap="binary")

fig.savefig(
    "res.png",
)

img = img[50:350, 50:350]


def foo(n: int, image: np.ndarray, ):
    lin = np.linspace(0, 180, n)
    radon_ = radon(img, lin)
    iradon_ = iradon(radon_, theta=lin)
    err = img - iradon_
    return image, radon_, iradon_, err

img1, radon1, iradon1, err1 = foo(10, img)
img2, radon2, iradon2, err2 = foo(30, img)
img3, radon3, iradon3, err3 = foo(100, img)
img4, radon4, iradon4, err4 = foo(1000, img)

ax[0,0].imshow(img1, cmap="binary_r")
ax[0,1].imshow(radon1, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[0,2].imshow(iradon1, cmap="binary_r")
ax[0,3].imshow(err1, cmap="binary")
ax[1,0].imshow(img2, cmap="binary_r")
ax[1,1].imshow(radon2, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[1,2].imshow(iradon2, cmap="binary_r")
ax[1,3].imshow(err3, cmap="binary")
ax[2,0].imshow(img3, cmap="binary_r")
ax[2,1].imshow(radon3, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[2,2].imshow(iradon3, cmap="binary_r")
ax[2,3].imshow(err3, cmap="binary")
ax[3,0].imshow(img4, cmap="binary_r")
ax[3,1].imshow(radon4, cmap="binary_r", aspect="auto", interpolation="nearest")
ax[3,2].imshow(iradon4, cmap="binary_r")
ax[3,3].imshow(err4, cmap="binary")

fig.savefig(
    "res1.png",
)