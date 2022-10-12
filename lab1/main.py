import matplotlib.pyplot as plt
import numpy as np
# Zadanie 1

fig, ax = plt.subplots(3,3, figsize=(10,10))

X = np. linspace(0, 4*np.pi, num=40)
y = np.sin(X)

ax[0,0].plot(X,y)
X1 = X[:,np.newaxis] * X[np.newaxis, :]
f1 = y[:,None] * y[None,:]
ax[0,1].imshow(f1, cmap="binary")
ax[0,1].set_title(f"min={np.round_(np.min(f1), decimals=3)}, max={np.round_(np.max(f1), decimals=3)}")

f2 = f1 - np.min(f1)
f2 /= np.max(f2)
ax[0,2].imshow(f2, cmap="binary")
ax[0,2].set_title(f"min={np.round_(np.min(f2), decimals=3)}, max={np.round_(np.max(f2), decimals=3)}")

# Zadanie 2
(L1, L2, L3) = [np.rint(f2*scaler) for scaler in [2**k-1 for k in [2,4,8]]]
ax[1,0].imshow(L1, cmap="binary")
ax[1,1].imshow(L2, cmap="binary")
ax[1,2].imshow(L3, cmap="binary")
ax[1,0].set_title("min=%.3f, max=%.f" % (np.min(L1), np.max(L1)))
ax[1,1].set_title("min=%.3f, max=%.3f" % (np.min(L2), np.max(L2)))
ax[1,2].set_title("min=%.3f, max=%.3f" % (np.min(L3), np.max(L3)))
# ax[1,1].set_title(f"min={np.round_(np.min(f2), decimals=3)}, max={np.round_(np.max(f2), decimals=3)}")
# ax[1,2].set_title(f"min={np.round_(np.min(f2), decimals=3)}, max={np.round_(np.max(f2), decimals=3)}")

fig.savefig("res.jpg")

#Zadanie 3 
n1 = np.random.normal(size=(40,40))
noised1 = n1 + f2
noised1
ax[2,0].imshow(noised1, cmap="binary")
ax[2,0].set_title("noise")

x1 = np.copy(noised1)
for n in range(50):
    x1 += f2 + np.random.normal(size=(40,40))


x2 = noised1
for n in range(1000):
    x2 += f2 + np.random.normal(size=(40,40))

x2 = x2 - np.min(x2)
x2 /= np.max(x2)

x1 = x1 - np.min(x1)
x1 /= np.max(x1)


ax[2,1].imshow(x1, cmap="binary")
ax[2,2].imshow(x2, cmap="binary")
ax[2,0].set_title("noise")
ax[2,0].set_title("noise")
ax[2,1].set_title("n=50")
ax[2,2].set_title("n=1000")
fig.savefig("result.jpg")