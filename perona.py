import numpy as np
import imutils
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as pyp


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

        if t == 0:
            hist, edges = np.histogram(np.abs((dn, ds, de, dw)), 256, density=True)
            bin_width = edges[1] - edges[0]
            # hist *= bin_width
            pyp.bar(range(256), hist)
            pyp.grid('on')
            pyp.show()
            cg = np.cumsum(hist)
            pyp.plot(cg)


        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                             lam * (f(dn, b) * dn + f(ds, b) * ds +
                                    f(de, b) * de + f(dw, b) * dw)
        im = im_new
    return im, np.sqrt(2 * lam * steps)


if __name__ == '__main__':
    img = imutils.imread('noisy-empire.png')
    print img
    diff = diffusion_filter(img, 40, lam=0.25)[0]
    imutils.imshow(diff)
    print diff

    # ani = anisotropic_diffusion_filter(img, 40, .9)[0]
    # imutils.imshow(ani)
    # print gaussian_filter(img, 4.4721)
    # imutils.imshow(img, title="Original Image")
    # imutils.imshow(diffusion_filter(img, 40, lam=0.25)[0], title="Diffusion")
    # imutils.imshow(anisotropic_diffusion_filter(img, 40, 0.1)[0], title="Perona-Malik")
    # imutils.imshow(gaussian_filter(img, 4.4721), title="Gaussian")
