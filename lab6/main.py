import copy
import numpy as np
import matplotlib.pyplot as plt
# from skimage import data
from skimage.draw import disk

def normalize(img):
    img = img - np.min(img)
    img /= np.max(img)
    return img

fig, ax = plt.subplots(3, 2, figsize=(10, 10))

image1 = plt.imread("image1.jpeg")
image2 = plt.imread("image2.jpg")
mean_img = np.mean(image1, axis=2)
normalized = normalize(mean_img)
bit_8_multiplier = 2**8 - 1 
normalized_8_bit_depth = np.rint(normalized * bit_8_multiplier)
ax[0, 0].imshow(normalized_8_bit_depth, cmap="binary_r")
ft = np.fft.fft2(normalized_8_bit_depth)
ft = np.fft.fftshift(ft)
ft_abs = np.abs(ft)
ft_log = np.log(ft_abs)


normalized_ft_log = normalize(ft_log)

normalized_ft_log_05 = copy.deepcopy(normalized_ft_log)
normalized_ft_log_08 = copy.deepcopy(normalized_ft_log)

normalized_ft_log_05[normalized_ft_log_05<=0.5] = 0
normalized_ft_log_05[normalized_ft_log_05>0.5] = 1

normalized_ft_log_08[normalized_ft_log_08<=0.8] = 0
normalized_ft_log_08[normalized_ft_log_08>0.8] = 1

print("Normalized threshold 0.5")
print(np.argwhere(normalized_ft_log_05))
print("Normalized threshold 0.8")
print(np.argwhere(normalized_ft_log_08))

ax[0,0].imshow(mean_img, cmap="gray")
ax[0,1].imshow(ft_log, cmap="gray")
ax[1,0].imshow(normalized_ft_log_05, cmap="gray")
ax[1,1].imshow(normalized_ft_log_08, cmap="gray")


# Zadanie 2

ft[75, :] = 0
ft[93, :] = 0
ft[131, :] = 0
ft[149, :] = 0

ft_abs_2 = np.abs(ft)
ft_log_2 = np.log(ft_abs_2)

ax[2,0].imshow(ft_log_2, cmap="gray")

def genereate_ift(ft):
    ift = np.fft.ifftshift(ft)
    ift = np.fft.ifft2(ift)
    return ift

ift = genereate_ift(ft).real
# normalized_ift = normalize(iffft)
# ifft_abs_2 = np.abs(ift)
# ifft_log_2 = np.log(ifft_abs_2)

# print(ifft_log_2)
normalized_ift = normalize(ift)
# ax[1,1].imshow(ift, cmap="gray")
ax[2,1].imshow(normalized_ift, cmap="gray")


fig.savefig(
    "res.png",
)

# Zadanie 3

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
image2 = plt.imread("image2.jpg")
print(image2.shape)

image2 = normalize(image2.astype(float))

ft__2 = np.fft.fft2(image2)
ft__2 = np.fft.fftshift(ft__2)
ft__2_abs = np.abs(ft__2)
ft__2_log = np.log(ft__2_abs)

normalized_ft_2_log = normalize(ft__2_log)

normalized_ft_2_log[normalized_ft_2_log<=0.7] = 0
normalized_ft_2_log[normalized_ft_2_log>0.7] = 1

print("Normalized threshold 0.7")
print(np.argwhere(normalized_ft_2_log))

print(image2)

rr, cc = disk((112, 320), 10)

ax[0, 0].imshow(image2, cmap="binary_r")
ax[0, 1].imshow(ft__2_log, cmap="binary_r")

ft__2 = np.fft.fft2(image2)
ft__2 = np.fft.fftshift(ft__2)
ft__2_abs = np.abs(ft__2)
ft__2_log = np.log(ft__2_abs)
rr, cc = disk((112, 320), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((172, 238), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((230, 150), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((174, 402), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((300, 238), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((350, 320), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((299, 401), 10)
ft__2_log[rr, cc] = 0
rr, cc = disk((232, 485), 10)
ft__2_log[rr, cc] = 0

ft__2_log[rr, cc] = 0

ax[1,0].imshow(ft__2_log, cmap="binary_r")

rr, cc = disk((112, 320), 10)
ft__2[rr, cc] = 0
rr, cc = disk((172, 238), 10)
ft__2[rr, cc] = 0
rr, cc = disk((230, 150), 10)
ft__2[rr, cc] = 0
rr, cc = disk((174, 402), 10)
ft__2[rr, cc] = 0
rr, cc = disk((300, 238), 10)
ft__2[rr, cc] = 0
rr, cc = disk((350, 320), 10)
ft__2[rr, cc] = 0
rr, cc = disk((299, 401), 10)
ft__2[rr, cc] = 0
rr, cc = disk((232, 485), 10)
ft__2[rr, cc] = 0

ift = genereate_ift(ft__2).real

ax[1,1].imshow(ift, cmap="binary_r")


fig.savefig(
    "res2.png",
)