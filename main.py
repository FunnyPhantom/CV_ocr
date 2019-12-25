import numpy as np
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt

img1 = io.imread("./data/1.jpg", as_gray=True)
img2 = io.imread("./data/2.jpg", as_gray=True)
img3 = io.imread("./data/3.jpg", as_gray=True)


def block_shaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


segment_img1 = block_shaped(img1, 70, 70)
segment_img2 = block_shaped(img2, 70, 70)
segment_img3 = block_shaped(img3, 70, 70)[0:70]

allData = np.concatenate((segment_img1, segment_img2, segment_img3), axis=0)

allData = allData[:, 5:65, 5:65]


def binerize(x):
    return 1 if x > 0.9 else 0


vectorized_binerize = np.vectorize(binerize)


allData = trans.resize(allData, (470, 28, 28))

allData = vectorized_binerize(allData)

np.save('duset_daram.npy', allData)

for i in range(10):
    io.imshow(allData[i])
    io.show()


for i in range(10):
    io.imshow(allData[i*10])
    io.show()


