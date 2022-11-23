import numpy as np
import matplotlib.pyplot as plt
from skimage import data

fig, ax = plt.subplots(3, 4, figsize=(10, 10))


n = 100
x = np.linspace(0, 11 * np.pi, n)
sin = np.sin(x)
img = sin[:, np.newaxis] * sin[np.newaxis, :]


def normalize(img):
    img = img - np.min(img)
    img /= np.max(img)
    return img


normalized = normalize(img)
bit_8_multiplier = 2**8 -1 
normalized_8_bit_depth = np.rint(normalized * bit_8_multiplier)
ax[0, 0].imshow(normalized_8_bit_depth, cmap="binary_r")


ft = np.fft.fft2(normalized_8_bit_depth)
ft = np.fft.fftshift(ft)

ft_abs = np.abs(ft)
ft_log = np.log(ft_abs)
print(ft_abs)
ax[0, 1].imshow(ft_abs, cmap="binary_r")
ax[0, 2].imshow(ft_log, cmap="binary_r")

# zadanie 2

lin = np.linspace(0, 11 * np.pi, 100)
x, y = np.meshgrid(lin, lin)


aplitudes = [0.4, 0.8, 1, 4, 10]
wavelengths = [0.5, 3, 6, 2.1, 8]
angles = [0.3 * np.pi, 1.7 * np.pi, 2 * np.pi, 5 * np.pi, 0.2 * np.pi]

outcome_array = np.zeros((100, 100, 5), dtype=float)
for dim, (amplitude, angle, wavelength) in enumerate(
    zip(aplitudes, angles, wavelengths)
):
    outcome_array[:, :, dim] = amplitude * np.sin(
        2 * np.pi * (x * np.cos(angle) + y * np.sin(angle)) * (1 / wavelength)
    )

print(outcome_array)

result_2 = outcome_array.sum(axis=2)
print(result_2.dtype)

ft2 = np.fft.fft2(result_2)
ft2 = np.fft.fftshift(ft2)
abs_ft2 = np.abs(ft2)
log_ft2 = np.log(abs_ft2)

ax[1, 0].imshow(result_2, cmap="binary_r")
ax[1, 1].imshow(abs_ft2, cmap="binary_r")
ax[1, 2].imshow(log_ft2, cmap="binary_r")

cam = data.camera()
ft_cam = np.fft.fft2(cam)
ft_cam = np.fft.fftshift(ft_cam)

ft_abs_cam = np.abs(ft_cam)
ft_log_cam = np.log(ft_abs_cam)

ax[2,0].imshow(cam, cmap="binary_r")
ax[2,1].imshow(ft_abs_cam, cmap="binary_r")
ax[2,2].imshow(ft_log_cam, cmap="binary_r")


# zadanie 3


def genereate_ift(ft):
    ift = np.fft.ifftshift(ft)
    ift = np.fft.ifft2(ift)
    return ift


def colored(ft):
    img_shape = (ft.shape[0], ft.shape[1], 3)
    res_3 = np.zeros(img_shape)
    res_3[:,:, 0] = normalize(genereate_ift(ft.real).real)
    res_3[:,:, 1] = normalize(genereate_ift(ft.imag * 1j).real)
    res_3[:,:, 2] = normalize(genereate_ift(ft))
    return res_3

x = colored(ft)
y = colored(ft2)
z = colored(ft_cam)

ax[0,3].imshow(x, cmap="binary")
ax[1,3].imshow(y, cmap="binary")
ax[2,3].imshow(z, cmap="binary")



fig.savefig(
    "res.png",
)
