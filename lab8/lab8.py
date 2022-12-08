from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.decomposition import PCA


fig, ax = plt.subplots(2, 3, figsize=(10, 6))
img_corrected = scipy.io.loadmat("SalinasA_corrected.mat")["salinasA_corrected"]


ch_10 = img_corrected[:, :, 10]
ch_100 = img_corrected[:, :, 100]
ch_200 = img_corrected[:, :, 200]
# img_1
ax[0, 0].imshow(ch_10, cmap="binary_r")
ax[0, 1].imshow(ch_100, cmap="binary_r")
ax[0, 2].imshow(ch_200, cmap="binary_r")

spectral_signature_10_10_y = img_corrected[10, 10, :]
spectral_signature_40_40_y = img_corrected[40, 40, :]
spectral_signature_80_80_y = img_corrected[80, 80, :]

ax[1, 0].plot(spectral_signature_10_10_y)
ax[1, 1].plot(spectral_signature_40_40_y)
ax[1, 2].plot(spectral_signature_80_80_y)
fig.savefig(
    "res_1.png",
)
img_shape = img_corrected.shape

def normalize(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    return img

image: np.ndarray = img_corrected.astype(float)

new_image = np.zeros((img_shape[0], img_shape[1], 3))
new_image[:, :, 0] = normalize(image[:, :, 4])
new_image[:, :, 1] = normalize(image[:, :, 12])
new_image[:, :, 2] = normalize(image[:, :, 26])

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

pca = PCA(3)
reshaped = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))

after_pca = pca.fit_transform(reshaped)
pca_image = after_pca.reshape((image.shape[0], image.shape[1], 3))
pca_image[:,:,0] = normalize(pca_image[:,:,0]) 
pca_image[:,:,1] = normalize(pca_image[:,:,1]) 
pca_image[:,:,2] = normalize(pca_image[:,:,2]) 
ax[0].imshow(new_image)
ax[1].imshow(pca_image)
fig.savefig(
    "res_2.png",
)

# img_raw = scipy.io.loadmat("/Users/hulewicz/Private/przetwarzanie_sygnalow/lab8/SalinasA_gt.mat")
labels = scipy.io.loadmat("/Users/hulewicz/Private/przetwarzanie_sygnalow/lab8/SalinasA_gt.mat")["salinasA_gt"]

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()

X = img_corrected.reshape(img_corrected.shape[0] * img_corrected.shape[1], img_corrected.shape[2])
y = labels.reshape(labels.shape[0]* labels.shape[1])
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)
X = X[y != 0]
y = y[y != 0]
print(y)
scores = []
for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
print(mean_score)
