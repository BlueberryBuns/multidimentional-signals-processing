import numpy as np
import matplotlib.pyplot as plt
from skimage.data import chelsea
from skimage.transform import AffineTransform, warp
from scipy import interpolate


def normalize(img):
    img = img - np.min(img)
    img /= np.max(img)
    return img


fig, ax = plt.subplots(4,2, figsize=(10,13))
chls = chelsea()
chlsea_array = np.array(chls, dtype=float)

ax[0,0].imshow(chls)
mean_cheslea = np.mean(chlsea_array, axis=2)
reduced_chelsea = mean_cheslea[::8, ::8]

normalized_chelsea = normalize(reduced_chelsea)
print(normalized_chelsea)
ax[0,1].imshow(normalized_chelsea, cmap="binary_r")
# print(reduced_chelsea)

ang = - np.pi / 12
affinie_matrix_1= np.eye(3,3)
affinie_matrix_1[0][0] = np.cos(ang)
affinie_matrix_1[1][1] = np.cos(ang)
affinie_matrix_1[0][1] = -np.sin(ang)
affinie_matrix_1[1][0] = np.sin(ang)
print(affinie_matrix_1)

transform_1 = AffineTransform(matrix = affinie_matrix_1)


affinie_matrix_2= np.eye(3,3)
affinie_matrix_2[0][1] = 0.5
transform_2 = AffineTransform(matrix = affinie_matrix_2)

img1 = warp(normalized_chelsea, transform_1)
img2 = warp(normalized_chelsea, transform_2)
ax[1,0].imshow(img1, cmap="binary_r")
ax[1,1].imshow(img2, cmap="binary_r")
# zadanie 2
def foo(img: np.ndarray):
    s = np.shape(img)
    x = np.linspace(0, s[1] * 8, s[1])
    y = np.linspace(0, s[0] * 8, s[0])
    f = interpolate.interp2d(x, y, img, kind="cubic")

    x_new = np.linspace(0, s[1] * 8, s[1] * 8)
    y_new = np.linspace(0, s[0] * 8, s[0] * 8)
    image_u = f(x_new, y_new)
    print(image_u[0:15, 0:15].round(1))
    return image_u

n_image_1 = foo(img1)
n_image_2 = foo(img2)
ax[2, 0].imshow(n_image_1, cmap="binary_r")
ax[2, 1].imshow(n_image_2, cmap="binary_r")

fig.savefig("result.jpg")
# import pdb; pdb.set_trace()
