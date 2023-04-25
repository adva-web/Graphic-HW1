import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


# pad the image to the size of the target image
def add_padding(im_src, im_tgt, im_mask, center):
    # Calculate row and column offsets
    offset_rows = center[0] - im_mask.shape[0] // 2
    offset_cols = center[1] - im_mask.shape[1] // 2
    # Update source
    src_resized = np.zeros((im_tgt.shape[0], im_tgt.shape[1], 3), dtype=np.int64)
    src_resized[offset_rows:-offset_rows, offset_cols:-offset_cols, :] = im_src
    # Update mask
    mask_resized = np.zeros((im_tgt.shape[0], im_tgt.shape[1]), dtype=np.int64)
    mask_resized[offset_rows:-offset_rows, offset_cols:-offset_cols] = im_mask

    return src_resized, mask_resized


# Aa suggested in https://hingxyu.medium.com/gradient-domain-fusion-using-poisson-blending-8a7dc1bbaa7b
# To prevent the inside mask area is in the border (edge case)
def add_img_border(im_src, im_tgt, im_mask):
    # add a 1-pixel symmetric buffer if the valid pixels in the mask are on the edge of the source image
    im_src = cv2.copyMakeBorder(im_src, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    im_tgt = cv2.copyMakeBorder(im_tgt, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    im_mask = cv2.copyMakeBorder(im_mask, 1, 1, 1, 1, cv2.BORDER_REFLECT)

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


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Add padding to images
    im_src, im_mask = add_padding(im_src, im_tgt, im_mask, center)
    im_src, im_tgt, im_mask = add_img_border(im_src, im_tgt, im_mask)

    # Get new dimensions for target
    rows, cols = im_tgt.shape[:2]

    # Flatten matrices
    mask_vector = im_mask.flatten()
    src_vectors = create_rgb_vectors(im_src)
    tgt_vectors = create_rgb_vectors(im_tgt)

    # Create
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

        tmp = spsolve(A, B)
        tmp = tmp.reshape((rows, cols))
        tmp[tmp > 255] = 255
        tmp[tmp < 0] = 0
        im_tgt[:, :, rgb] = tmp.astype('uint8')

    im_blend = im_tgt[1:-1, 1:-1, :]
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana2.bmp', help='mask file path')
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

    center = (int(im_tgt.shape[0] / 2), int(im_tgt.shape[1] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
