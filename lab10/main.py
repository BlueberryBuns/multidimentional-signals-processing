from matplotlib import pyplot as plt
from skimage.data import chelsea
from skimage.segmentation import slic, watershed, quickshift
import numpy as np
from skimage.color import label2rgb
from skimage.feature import canny


def normalize(img):
    img = img.astype(float)
    img = img - np.min(img)
    img /= np.max(img)
    return img.astype(float)


fig, ax = plt.subplots(3, 3, figsize=(10, 10))

img = chelsea()
# img = normalize(img)

img_grey = img.mean(axis=2)

slic_img = slic(img)
segments_slic = np.unique(slic_img)[-1]
watershed_img = watershed(img_grey)
segments_watershed = np.unique(watershed_img)[-1]
quickshift_img = quickshift(img)
segments_quickshift = np.unique(quickshift_img)[-1]


ax[0, 0].imshow(slic_img, cmap="twilight")
ax[0, 0].set_title(f"Segments {segments_slic}")
ax[1, 0].imshow(watershed_img, cmap="twilight")
ax[1, 0].set_title(f"Segments {segments_watershed}")

ax[2, 0].imshow(quickshift_img, cmap="twilight")
ax[2, 0].set_title(f"Segments {segments_quickshift}")


slic_img_avg = label2rgb(slic_img, img, kind="avg")
watershed_img_avg = label2rgb(watershed_img, img, kind="avg")
quickshift_img_avg = label2rgb(quickshift_img, img, kind="avg")

slic_img_kind = label2rgb(slic_img, img, kind="overlay")
watershed_img_kind = label2rgb(watershed_img, img, kind="overlay")
quickshift_img_kind = label2rgb(quickshift_img, img, kind="overlay")

ax[0, 1].imshow(slic_img_kind, cmap="twilight")
ax[1, 1].imshow(watershed_img_kind, cmap="twilight")
ax[2, 1].imshow(quickshift_img_kind, cmap="twilight")
ax[0, 2].imshow(slic_img_avg, cmap="twilight")
ax[1, 2].imshow(watershed_img_avg, cmap="twilight")
ax[2, 2].imshow(quickshift_img_avg, cmap="twilight")

canny_mask = canny(img_grey, sigma=3.0)
slic_img_avg[canny_mask] = 1
watershed_img_avg[canny_mask] = 1
quickshift_img_avg[canny_mask] = 1


ax[0, 1].imshow(slic_img_kind, cmap="twilight")
ax[1, 1].imshow(watershed_img_kind, cmap="twilight")
ax[2, 1].imshow(quickshift_img_kind, cmap="twilight")
ax[0, 2].imshow(slic_img_avg)
ax[1, 2].imshow(watershed_img_avg)
ax[2, 2].imshow(quickshift_img_avg)
fig.savefig("result.jpg")
