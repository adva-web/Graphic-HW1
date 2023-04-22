import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


# pad the image to the size of the target image
def add_paddind(im_tgt, img):
    pad_top = (im_tgt.shape[0] - img.shape[0]) // 2
    pad_bottom = im_tgt.shape[0] - img.shape[0] - pad_top
    pad_left = (im_tgt.shape[1] - img.shape[1]) // 2
    pad_right = im_tgt.shape[1] - img.shape[1] - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


def get_gradient(img):
    kernel_x = np.array([[0, 0, 0],
                         [0, -1, 1],
                         [0, 0, 0]])
    kernel_y = np.array([[0, 0, 0],
                         [0, -1, 0],
                         [0, 1, 0]])
    grad_x = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_32F, kernel_y)
    return grad_x, grad_y


def get_laplacian(grad_x, grad_y):
    kernel_x = np.array([[0, 0, 0],
                         [-1, 1, 0],
                         [0, 0, 0]])
    kernel_y = np.array([[0, -1, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
    lap_x = cv2.filter2D(grad_x, cv2.CV_32F, kernel_x)
    lap_y = cv2.filter2D(grad_y, cv2.CV_32F, kernel_y)
    return lap_x + lap_y


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-4)
    mat_D.setdiag(1, -1)
    mat_D.setdiag(1, 1)
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    mat_A.setdiag(1, -m)
    mat_A.setdiag(1, m)

    return mat_A.tocsc()


def poisson_blend(source, target, mask, center):
    y_range, x_range = target.shape[:-1]  # height, width

    offset_rows = center[1] - im_mask.shape[0] // 2
    offset_cols = center[0] - im_mask.shape[1] // 2
    shift = offset_rows, offset_cols

    M = np.float32([[1, 0, shift[0]],
                    [0, 1, shift[1]]])
    source = cv2.warpAffine(source, M, (x_range, y_range))
    mask = cv2.warpAffine(mask, M, (x_range, y_range))

    s_grad_x, s_grad_y = get_gradient(source)
    t_grad_x, t_grad_y = get_gradient(target)

    for c in range(target.shape[2]):
        for y in range(y_range):
            for x in range(x_range):
                if mask[y, x] != 0:
                    if abs(t_grad_x[y, x, c]) + abs(t_grad_y[y, x, c]) < abs(s_grad_x[y, x, c]) + abs(
                            s_grad_y[y, x, c]):
                        t_grad_x[y, x, c] = s_grad_x[y, x, c]
                        t_grad_y[y, x, c] = s_grad_y[y, x, c]
                        # mixed gradient 按梯度绝对值分配 gradient

    lap = get_laplacian(t_grad_x, t_grad_y)
    # laplacian expression of result image

    mat_A = laplacian_matrix(y_range, x_range)

    # 修改边界点
    lap[mask == 0] = target[mask == 0]
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k - 1] = mat_A[k, k + 1] = 0
                mat_A[k, k - x_range] = mat_A[k, k + x_range] = 0

    res = np.zeros(source.shape)
    mask_flat = mask.flatten()
    for channel in range(target.shape[2]):
        mat_B = lap[:, :, channel].flatten()

        tmp = scipy.sparse.linalg.spsolve(mat_A, mat_B)
        tmp = tmp.reshape((y_range, x_range))
        tmp[tmp > 255] = 255
        tmp[tmp < 0] = 0
        res[:, :, channel] = tmp.astype('uint8')

    return res


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
