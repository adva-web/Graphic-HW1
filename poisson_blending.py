import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


# Pad the image to the size of the target image
def add_padding(im_tgt, img):
    pad_top = (im_tgt.shape[0] - img.shape[0]) // 2
    pad_bottom = im_tgt.shape[0] - img.shape[0] - pad_top
    pad_left = (im_tgt.shape[1] - img.shape[1]) // 2
    pad_right = im_tgt.shape[1] - img.shape[1] - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


# As suggested in https://hingxyu.medium.com/gradient-domain-fusion-using-poisson-blending-8a7dc1bbaa7b
# To prevent the inside mask area is in the border (edge case)
def add_img_border(im_src, im_tgt, im_mask):
    # Add a 1-pixel symmetric buffer if the valid pixels in the mask are on the edge of the source image
    im_src = cv2.copyMakeBorder(im_src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    im_tgt = cv2.copyMakeBorder(im_tgt, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    im_mask = cv2.copyMakeBorder(im_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    return im_src, im_tgt, im_mask


def calculate_laplacian_matrix(rows, cols):
    D = scipy.sparse.lil_matrix((cols, cols))
    D.setdiag(4)
    D.setdiag(-1, -1)
    D.setdiag(-1, 1)
    A = scipy.sparse.block_diag([D] * rows).tolil()
    A.setdiag(-1, 1 * cols)
    A.setdiag(-1, -1 * cols)
    return A, A.tocsc()


def create_vector(img, rgb):
    return img[:, :, rgb].flatten()


def create_rgb_vectors(img):
    return [create_vector(img, rgb) for rgb in range(img.shape[2])]


def solve_poisson_equation(A, B, rows, cols):
    tmp = spsolve(A, B)
    tmp = tmp.reshape((rows, cols))
    tmp[tmp > 255] = 255
    tmp[tmp < 0] = 0
    return tmp.astype('uint8')


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Add padding to images
    im_src = add_padding(im_tgt, im_src)
    im_mask = add_padding(im_tgt, im_mask)
    im_src, im_tgt, im_mask = add_img_border(im_src, im_tgt, im_mask)

    # Get the new dimensions of target image
    rows, cols = im_tgt.shape[:2]

    # Flatten matrices
    mask_vector = im_mask.flatten()
    src_vectors = create_rgb_vectors(im_src)
    tgt_vectors = create_rgb_vectors(im_tgt)

    # Create laplacian matrix
    A, laplacian = calculate_laplacian_matrix(rows, cols)

    # Outside the blending part (according to mask)
    # we want to create I matrix
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            # The value of blending part im mask is 1 and outside is 0
            if im_mask[y, x] == 0:
                k = x + y * cols
                A[k, k] = 1
                A[k, k - 1] = A[k, k + 1] = 0
                A[k, k - cols] = A[k, k + cols] = 0
    A = A.tocsc()

    for rgb in range(im_src.shape[2]):
        B = laplacian.dot(src_vectors[rgb])

        # Outside the mask we put target pixels as it is
        outside_mask_pixels = np.where(mask_vector == 0)
        B[outside_mask_pixels] = tgt_vectors[rgb][outside_mask_pixels]

        im_tgt[:, :, rgb] = solve_poisson_equation(A, B, rows, cols)

    im_blend = im_tgt[1:-1, 1:-1, :]
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/stone2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/stone2.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/grass_mountains.jpeg', help='mask file path')
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

    center = (int(im_tgt.shape[0] / 2), int(im_tgt.shape[1] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
