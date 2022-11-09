from matplotlib import pyplot as plt
import numpy as np
from skimage.data import chelsea


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    img = img - np.min(img)
    img /= np.max(img)
    return img

X_sin = np.linspace(0, 2*np.pi, 256)
y_sin = np.sin(X_sin)

y_2 = y_sin + 1
y_2 /= 2
y_2 *= 255
y_2 = np.rint(y_2)
y_2 = y_2.astype(int)
print(y_2)

def gamma_tranformation(img: np.ndarray, gamma: float) -> np.ndarray:
    img = normalize(img)
    tmp = img ** (1/ gamma)
    tmp = tmp / tmp.max()
    tmp *= 255
    return tmp.astype(int)


fig, ax = plt.subplots(6,3, figsize=(10,13))
chls = chelsea()
chlsea_array = np.array(chls, dtype=int)
X = np.arange(0,256, dtype=int)
lut_identity = np.arange(0,256, dtype=int)
lut_neg = lut_identity[::-1]
lut_threshhold = np.zeros(256, dtype=int)
lut_threshhold[50:200] = lut_identity[-1]
lut_sin = y_2
lut_gamma = gamma_tranformation(lut_identity, 0.3)
lut_gamma_3 = gamma_tranformation(lut_identity, 3)

img_identity = lut_identity[chls]
ax[0,0].plot(X,lut_identity)
ax[1,0].plot(X,lut_neg)
ax[2,0].plot(X,lut_threshhold)
ax[3,0].plot(X,lut_sin)
ax[4,0].plot(X,lut_gamma)
ax[5,0].plot(X,lut_gamma_3)
print(lut_identity[chls].dtype)
print(lut_neg[chls].dtype)
print(lut_threshhold[chls].dtype)
print(lut_sin[chls].dtype)
print(lut_gamma[chls].dtype)
print(lut_identity[chls].dtype)
ax[0,1].imshow(lut_identity[chls])
ax[1,1].imshow(lut_neg[chls])
ax[2,1].imshow(lut_threshhold[chls])
ax[3,1].imshow(lut_sin[chls])
ax[4,1].imshow(lut_gamma[chls])
ax[5,1].imshow(lut_gamma_3[chls])


# zadanie 2
def gent_hist_data(img, idx_x, idx_y, plot):
    s = np.sum(np.unique(chls, return_counts=True)[1])
    chls_r = img[:,:,0]
    chls_g = img[:,:,1]
    chls_b = img[:,:,2]
    a = np.unique(chls_r, return_counts=True)

    b = np.unique(chls_g, return_counts=True)
    c = np.unique(chls_b, return_counts=True)
    plot[idx_x,idx_y].plot(a[0], a[1]/a[1].sum(), color='r')
    plot[idx_x,idx_y].plot(b[0], b[1]/b[1].sum(), color='g')
    plot[idx_x,idx_y].plot(c[0], c[1]/c[1].sum(), color='b')



# ax[0,2].plot(a[0], (a[1]/s)
gent_hist_data(chls, 0,2, ax)
gent_hist_data(lut_neg[chls], 1,2, ax)
gent_hist_data(lut_threshhold[chls], 2,2, ax)
gent_hist_data(lut_sin[chls], 3,2, ax)
gent_hist_data(lut_gamma[chls], 4,2, ax)
gent_hist_data(lut_gamma_3[chls], 5,2, ax)
# ax[1,2].plot()
# ax[2,2].plot()
# ax[3,2].plot()
# ax[4,2].plot()
# ax[5,2].plot()

plt.tight_layout()
fig.savefig("result.jpg")