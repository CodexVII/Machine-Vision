import numpy as np
import imutils
from scipy.ndimage import gaussian_filter


def diffusion_filter(im, steps, lam=0.25):
    im_new = np.zeros(im.shape, dtype=im.dtype)
    for t in range(steps):
        # im_new[1:-1, 1:-1] = \
        #     im[1:-1, 1:-1] + \
        #     lam * (im[1:-1, 2:] +
        #            im[1:-1, :-2] +
        #            im[2:, 1:-1] +
        #            im[:-2, 1:-1] -
        #            4 * im[1:-1, 1:-1])
        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]

        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                             lam * (dn + ds + de + dw)

        im = im_new
    return im, np.sqrt(2 * lam * steps)


def f(direction, b):
    return np.exp(-(np.power(direction, 2.0) / np.power(b, 2.0)))


def anisotropic_diffusion_filter(im, steps, b, lam=0.25):
    im_new = np.zeros(im.shape, dtype=im.dtype)

    for t in range(steps):
        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]

        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                             lam * (f(dn, b) * dn + f(ds, b) * ds +
                                    f(de, b) * de + f(dw, b) * dw)
        im = im_new
    return im, np.sqrt(2 * lam * steps)


if __name__ == '__main__':
    img = imutils.imread('noisy-empire.png')
    imutils.imshow(img, title="Original Image")
    imutils.imshow(diffusion_filter(img, 40, lam=0.25)[0], title="Diffusion")
    imutils.imshow(anisotropic_diffusion_filter(img, 40, 20)[0], title="Perona-Malik")
    imutils.imshow(gaussian_filter(img, 4.4721), title="Gaussian")
