import numpy as np
import imutils
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def diffusion_filter(im, steps, lam=0.25):
    im_new = np.zeros(im.shape, dtype=im.dtype)
    for t in range(steps):
        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]

        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                             lam * (dn + ds + de + dw)
        im = im_new
    return im, np.sqrt(2 * lam * steps)


# high-constrast edges
def f(gradient, b):
    return np.exp(-(np.power(gradient, 2.0) / np.power(b, 2.0)))


b_history = []


def ff(gradient, b):
    if b == 0:
        return 0
    return 1.0 / (1.0 + np.power((np.abs(gradient) / b), 2.0))


def anisotropic_diffusion_filter(im, steps, type=0, lam=0.25):
    im_new = np.zeros(im.shape, dtype=im.dtype)

    for t in range(steps):
        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]

        b = make_b(dn, ds, de, dw, .9)
        b_history.append(b)
        # if t == 0:
        #     hist, edges = np.histogram(np.abs((dn, ds, de, dw)), 256, density=True)
        #     bin_width = edges[1] - edges[0]
        #     hist *= bin_width
        #     plt.bar(range(256), hist)
        #     plt.grid('on')
        #     plt.show()
        #     cg = np.cumsum(hist)
        #     plt.plot(cg)
        if type == 0:
            im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                                 lam * (f(dn, b) * dn + f(ds, b) * ds +
                                        f(de, b) * de + f(dw, b) * dw)
        elif type == 1:
            im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                                 lam * (ff(dn, b) * dn + ff(ds, b) * ds +
                                        ff(de, b) * de + ff(dw, b) * dw)
        im = im_new
    return im, np.sqrt(2 * lam * steps)


def make_b(dn, ds, de, dw, C, H_BINS=50):
    "Per-cycle calculation of b for a given C"
    gm = np.fabs((dn, ds, de, dw))
    hist, edges, = np.histogram(gm, H_BINS, density=True)

    bin_width = edges[1] - edges[0]
    hist *= bin_width

    acc, b = hist[0], 0.0
    for j in range(1, len(hist)):
        if acc > C: break
        acc, b = acc + hist[j], b + bin_width
    return b


if __name__ == '__main__':
    # noisy-empire
    print "Reading noisy-empire.."
    building = imutils.imread('noisy-empire.png').astype("float32")
    building /= building.max()  # normalise
    print "Done."

    print "Performing filters.."
    ani = anisotropic_diffusion_filter(building, 40, type=0)[0]
    ani2 = anisotropic_diffusion_filter(building, 40, type=1)[0]
    print "Done."

    # noisy-rect
    print "Reading noisy-rect.."
    box = imutils.imread('noisy_rect.png').astype("float32")
    box /= box.max()    # normalise
    print "Done."

    print "Performing filters.."
    box_ani = anisotropic_diffusion_filter(box, 40, type=0)[0]
    box_ani2 = anisotropic_diffusion_filter(box, 40, type=1)[0]
    print "Done."

    # plot the things
    plt.subplot(131)
    plt.imshow(ani, cmap='gray')
    plt.title("First one")

    plt.subplot(132)
    plt.imshow(building, cmap='gray')
    plt.title("Original")

    plt.subplot(133)
    plt.imshow(ani2, cmap='gray')
    plt.title("Second one")
    plt.show()
    plt.savefig("Empire.png")


    # plot the things
    plt.subplot(131)
    plt.imshow(box_ani, cmap='gray')
    plt.title("First one")

    plt.subplot(132)
    plt.imshow(box, cmap='gray')
    plt.title("Original")

    plt.subplot(133)
    plt.imshow(box_ani2, cmap='gray')
    plt.title("Second one")
    plt.show()
    plt.savefig("Rect.png")


    # plot all the things
    plt.subplot(231)
    plt.imshow(ani, cmap='gray')
    plt.title("First one")

    plt.subplot(232)
    plt.imshow(building, cmap='gray')
    plt.title("Original")

    plt.subplot(233)
    plt.imshow(ani2, cmap='gray')
    plt.title("Second one")

    plt.subplot(234)
    plt.imshow(box_ani, cmap='gray')
    plt.title("First one")

    plt.subplot(235)
    plt.imshow(box, cmap='gray')
    plt.title("Original")

    plt.subplot(236)
    plt.imshow(box_ani2, cmap='gray')
    plt.title("Second one")
    plt.show()
    plt.savefig("Everything.png")