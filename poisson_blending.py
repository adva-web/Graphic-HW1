import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

remove_buffer = False


# pad the image to the size of the target image
def add_paddind(im_tgt, img):
    pad_top = (im_tgt.shape[0] - img.shape[0]) // 2
    pad_bottom = im_tgt.shape[0] - img.shape[0] - pad_top
    pad_left = (im_tgt.shape[1] - img.shape[1]) // 2
    pad_right = im_tgt.shape[1] - img.shape[1] - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


#TODO: if this function is not useful remove it
def img_translation(im_src, im_tgt, im_mask, center):
    global remove_buffer

    # calculate row and column offsets for translation
    offset_rows = center[1] - im_mask.shape[0] // 2
    offset_cols = center[0] - im_mask.shape[1] // 2

    # roll the images by the row and column offsets
    im_mask = np.roll(im_mask, (offset_rows, offset_cols), axis=(0, 1))
    im_src = np.roll(im_src, (offset_rows, offset_cols), axis=(0, 1))
    im_tgt = np.roll(im_tgt, (offset_rows, offset_cols), axis=(0, 1))

    # add a 1-pixel symmetric buffer if necessary
    if np.any(im_mask[0, :] == 255) or np.any(im_mask[-1, :] == 255) or \
            np.any(im_mask[:, 0] == 255) or np.any(im_mask[:, -1] == 255):
        im_mask = cv2.copyMakeBorder(im_mask, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        im_src = cv2.copyMakeBorder(im_src, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        im_tgt = cv2.copyMakeBorder(im_tgt, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        remove_buffer = True

    return im_src, im_tgt, im_mask


def calculate_gradient(img):
    rows_kernel = np.array([[0, 0, 0],
                            [0, -1, 1],
                            [0, 0, 0]])
    cols_kernel = np.array([[0, 0, 0],
                            [0, -1, 0],
                            [0, 1, 0]])
    rows_gradient = cv2.filter2D(img, cv2.CV_32F, rows_kernel, cv2.BORDER_REFLECT)
    cols_gradient = cv2.filter2D(img, cv2.CV_32F, cols_kernel, cv2.BORDER_REFLECT)
    return rows_gradient, cols_gradient


def calculate_norm(gradient, row, col, rgb):
    return abs(gradient[0][row, col, rgb]) + abs(gradient[1][row, col, rgb])


def calculate_laplacian(tgt_gradient):
    rows_kernel = np.array([[0, 0, 0],
                            [-1, 1, 0],
                            [0, 0, 0]])
    cols_kernel = np.array([[0, -1, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
    laplacian_rows = cv2.filter2D(tgt_gradient[0], cv2.CV_32F, rows_kernel, cv2.BORDER_REFLECT)
    laplacian_cols = cv2.filter2D(tgt_gradient[1], cv2.CV_32F, cols_kernel, cv2.BORDER_REFLECT)
    return laplacian_rows + laplacian_cols


def calculate_laplacian_matrix(tgt_height, tgt_width):
    # TODO: maybe should multiply by -1, in medium tgt_width should be tgt_height
    fl = scipy.sparse.lil_matrix((tgt_height, tgt_height))
    fl.setdiag(4)
    fl.setdiag(-1, -1)
    fl.setdiag(-1, 1)
    A = scipy.sparse.block_diag([fl] * tgt_width).tolil()
    A.setdiag(-1, 1 * tgt_height)
    A.setdiag(-1, -1 * tgt_height)
    return A


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    rows, cols = im_tgt.shape[:2]

    # calculate row and column offsets
    offset_rows = center[1] - im_mask.shape[0] // 2
    offset_cols = center[0] - im_mask.shape[1] // 2

    im_src = add_paddind(im_tgt, im_src)
    im_mask = add_paddind(im_tgt, im_mask)

    src_gradient = calculate_gradient(im_src)
    tgt_gradient = calculate_gradient(im_tgt)

    # apply the same gradient from the source image in the target image
    mask_indices = np.nonzero(im_mask)
    for row, col in zip(*mask_indices):
        for rgb in range(im_tgt.shape[2]):
            src_gradient_norm = calculate_norm(src_gradient, row, col, rgb)
            tgt_gradient_norm = calculate_norm(tgt_gradient, row, col, rgb)
            if tgt_gradient_norm < src_gradient_norm:
                tgt_gradient[0][row, col, rgb] = src_gradient[0][row, col, rgb]
                tgt_gradient[1][row, col, rgb] = src_gradient[1][row, col, rgb]

    laplacian = calculate_laplacian(tgt_gradient)

    A = calculate_laplacian_matrix(rows, cols)

    laplacian[im_mask == 0] = im_tgt[im_mask == 0]

    A_copy = A.copy()
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if im_mask[y, x] == 0:
                k = x + y * cols
                A_copy[k, k] = 1
                A_copy[k, k - 1] = A_copy[k, k + 1] = 0
                A_copy[k, k - cols] = A_copy[k, k + cols] = 0

    res = np.zeros(im_src.shape)
    for channel in range(im_tgt.shape[2]):
        B = laplacian[:, :, channel].flatten()

        tmp = scipy.sparse.linalg.spsolve(A_copy.tocsc(), B)
        tmp = tmp.reshape((rows, cols))
        tmp[tmp > 255] = 255
        tmp[tmp < 0] = 0
        res[:, :, channel] = tmp.astype('uint8')

    return res

    im_blend = im_tgt
    return im_src


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    #TODO: maybe replace to center = (int(im_tgt.shape[0] / 2), int(im_tgt.shape[1] / 2))
    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
