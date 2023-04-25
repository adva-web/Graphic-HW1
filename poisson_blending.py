import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


# TODO: remove if not necessary
# pad the image to the size of the target image
def add_padding(im_tgt, img):
    pad_top = (im_tgt.shape[0] - img.shape[0]) // 2
    pad_bottom = im_tgt.shape[0] - img.shape[0] - pad_top
    pad_left = (im_tgt.shape[1] - img.shape[1]) // 2
    pad_right = im_tgt.shape[1] - img.shape[1] - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


def add_img_border(im_src, im_tgt, im_mask):
    # add a 1-pixel symmetric buffer if the valid pixels in the mask are on the edge of the source image
    im_src = cv2.copyMakeBorder(im_src, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    im_tgt = cv2.copyMakeBorder(im_tgt, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    im_mask = cv2.copyMakeBorder(im_mask, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    return im_src, im_tgt, im_mask


# def calculate_gradient(img):
#     rows_kernel = np.array([[0, 0, 0],
#                             [0, -1, 1],
#                             [0, 0, 0]])
#     cols_kernel = np.array([[0, 0, 0],
#                             [0, -1, 0],
#                             [0, 1, 0]])
#     rows_gradient = cv2.filter2D(img, cv2.CV_32F, rows_kernel)
#     cols_gradient = cv2.filter2D(img, cv2.CV_32F, cols_kernel)
#     dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     print(rows_gradient.shape, dx.shape, np.sum(rows_gradient != dx))
#     print(cols_gradient.shape, dy.shape, np.sum(cols_gradient != dy))
#     return rows_gradient, cols_gradient


# def calculate_norm(gradient, row, col, rgb):
#     return abs(gradient[0][row, col, rgb]) + abs(gradient[1][row, col, rgb])


# def calculate_laplacian(tgt_gradient):
#     rows_kernel = np.array([[0, 0, 0],
#                             [-1, 1, 0],
#                             [0, 0, 0]])
#     cols_kernel = np.array([[0, -1, 0],
#                             [0, 1, 0],
#                             [0, 0, 0]])
#     laplacian_rows = cv2.filter2D(tgt_gradient[0], cv2.CV_32F, rows_kernel, cv2.BORDER_REFLECT)
#     laplacian_cols = cv2.filter2D(tgt_gradient[1], cv2.CV_32F, cols_kernel, cv2.BORDER_REFLECT)
#     return laplacian_rows + laplacian_cols


def calculate_laplacian_matrix(tgt_height, tgt_width):
    D = scipy.sparse.lil_matrix((tgt_height, tgt_height))
    D.setdiag(4)
    D.setdiag(-1, -1)
    D.setdiag(-1, 1)
    A = scipy.sparse.block_diag([D] * tgt_width).tolil()
    A.setdiag(-1, 1 * tgt_height)
    A.setdiag(-1, -1 * tgt_height)
    return A, A.tocsc()


def create_vector(img, rgb):
    return img[:, :, rgb].flatten()


def create_rgb_vectors(img):
    return [create_vector(img, rgb) for rgb in range(img.shape[2])]


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    rows, cols = im_tgt.shape[:2]

    # Calculate row and column offsets
    offset_rows = center[1] - im_mask.shape[0] // 2
    offset_cols = center[0] - im_mask.shape[1] // 2

    # im_src = add_padding(im_tgt, im_src)
    # im_mask = add_padding(im_tgt, im_mask)

    # Add padding to source and mask to size of target
    M = np.float32([[1, 0, offset_cols], [0, 1, offset_rows]])
    im_src = cv2.warpAffine(im_src, M, (cols, rows))
    im_mask = cv2.warpAffine(im_mask, M, (cols, rows))

    im_src, im_tgt, im_mask = add_img_border(im_src, im_tgt, im_mask)

    # Convert the mask's value to {0, 1} (0->0, 255->1)
    im_mask[im_mask != 0] = 1

    # Flatten matrices
    mask_vector = im_mask.flatten()
    src_vectors = create_rgb_vectors(im_src)
    tgt_vectors = create_rgb_vectors(im_tgt)

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
    print(laplacian.shape, src_vectors[1].shape)

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

    cv2.imshow('mask', im_mask)
    cv2.imshow('target', im_tgt)
    cv2.imshow('source', im_src)
    #
    # src_gradient = calculate_gradient(im_src)
    # tgt_gradient = calculate_gradient(im_tgt)
    #
    # # apply the same gradient from the source image in the target image
    # mask_indices = np.nonzero(im_mask)
    # for row, col in zip(*mask_indices):
    #     for rgb in range(im_tgt.shape[2]):
    #         src_gradient_norm = calculate_norm(src_gradient, row, col, rgb)
    #         tgt_gradient_norm = calculate_norm(tgt_gradient, row, col, rgb)
    #         if tgt_gradient_norm < src_gradient_norm:
    #             tgt_gradient[0][row, col, rgb] = src_gradient[0][row, col, rgb]
    #             tgt_gradient[1][row, col, rgb] = src_gradient[1][row, col, rgb]
    #
    # laplacian = calculate_laplacian(tgt_gradient)
    #
    # A = calculate_laplacian_matrix(rows, cols)
    #
    # laplacian[im_mask == 0] = im_tgt[im_mask == 0]
    #
    # A_copy = A.copy()
    # for y in range(1, rows - 1):
    #     for x in range(1, cols - 1):
    #         if im_mask[y, x] == 0:
    #             k = x + y * cols
    #             A_copy[k, k] = 1
    #             A_copy[k, k - 1] = A_copy[k, k + 1] = 0
    #             A_copy[k, k - cols] = A_copy[k, k + cols] = 0
    #
    # res = np.zeros(im_src.shape)
    # for channel in range(im_tgt.shape[2]):
    #     B = laplacian[:, :, channel].flatten()
    #
    #     tmp = scipy.sparse.linalg.spsolve(A_copy.tocsc(), B)
    #     tmp = tmp.reshape((rows, cols))
    #     tmp[tmp > 255] = 255
    #     tmp[tmp < 0] = 0
    #     res[:, :, channel] = tmp.astype('uint8')
    #
    # return res

    im_blend = im_tgt
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

    #TODO: maybe replace to center = (int(im_tgt.shape[0] / 2), int(im_tgt.shape[1] / 2))
    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
