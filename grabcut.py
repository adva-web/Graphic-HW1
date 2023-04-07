import numpy as np
import cv2
import argparse
import igraph
from sklearn.mixture import GaussianMixture as GMM

#############
# CONSTANTS #
#############
GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

# The constant gamma was obtained as 50 by optimizing performance against ground truth over a training set of 15 images.
# P.2 “GrabCut”
GAMMA = 50

####################
# GLOBAL VARIABLES #
####################
beta = 0
rows = 0
columns = 0
n_comp = 5
k = 0
weight_left = np.empty(0)
weight_up = np.empty(0)
weight_upleft = np.empty(0)
weight_upright = np.empty(0)


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    global beta
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Initialize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    initialize_params(img)

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    # TODO: should be 1000, n_iter == num_iter?
    num_iters = 1
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


# The constant beta is chosen to be: (according to “GrabCut” document formula (5))
# Beta = 1/(2*<(z_m-z_n)^2>)
# Where <> denotes expectation over an image sample.
def calculate_beta(left, up, upleft, upright):
    # Calculate sum of squared differences for each subarray
    left_sum = np.sum(np.square(left))
    up_sum = np.sum(np.square(up))
    upleft_sum = np.sum(np.square(upleft))
    upright_sum = np.sum(np.square(upright))

    # Calculate total sum of squared differences
    total_sum = left_sum + up_sum + upleft_sum + upright_sum

    # TODO: !!!!!!to check if we need to add or remove here somthing!!!!!!
    global beta, rows, columns
    # Each internal pixel has 4 neighbors (left, up, upleft, upright)
    num_of_neighbors = 4 * (columns - 2) * (rows - 1)
    # First column has 2 neighbors (up, upright) and last column has 3 neighbors (left, up, upleft)
    num_of_neighbors += 5 * (rows - 1)
    # First row has 1 neighbor (left)
    num_of_neighbors += columns - 1
    normalized_total_sum = total_sum / num_of_neighbors
    beta = 1 / (2 * normalized_total_sum)


# Calculate edge weight according to formula (1) in "Implementing GrabCut" document
# N(m,n) = (50/dist(m,n))*exp(-beta*||z_m-z_n||^2) -->  GAMMA=50
# Inspired by formula (11) in the original “GrabCut” document
def weight(dist_neighbors_mat, dist):
    global beta
    return (GAMMA/dist) * np.exp(-beta * np.sum(np.square(dist_neighbors_mat), axis=2))


def calculate_weights(left, up, upleft, upright):
    global weight_left, weight_up, weight_upleft, weight_upright
    diag_dist = np.sqrt(2)
    straight_dist = 1

    weight_left = weight(left, straight_dist)
    weight_up = weight(up, straight_dist)
    weight_upleft = weight(upleft, diag_dist)
    weight_upright = weight(upright, diag_dist)


# Calculate for each pixel the difference to its 4 direct neighbors
def calculate_dist_neighbors_matrix(img):
    # Difference between columns (col i+1 to col i)
    dist_left_pixels = np.diff(img, axis=1)
    # Difference between rows (row i+1 to row i)
    dist_up_pixels = np.diff(img, axis=0)
    # Difference between diagonal cells
    dist_upleft_pixels = img[1:, 1:] - img[:-1, :-1]
    dist_upright_pixels = img[1:, :-1] - img[:-1, 1:]
    return dist_left_pixels, dist_up_pixels, dist_upleft_pixels, dist_upright_pixels


def initialize_params(img):
    global rows, columns
    rows = img.shape[0]
    columns = img.shape[1]
    left, up, upleft, upright = calculate_dist_neighbors_matrix(img)
    calculate_beta(left, up, upleft, upright)
    calculate_weights(left, up, upleft, upright)


# Returns the background pixels and foreground pixels of image according to mask
def split_bg_fg_pixels(mask):
    bg_pixels = ((mask == GC_BGD) | (mask == GC_PR_BGD)).nonzero()
    fg_pixels = ((mask == GC_FGD) | (mask == GC_PR_FGD)).nonzero()
    return bg_pixels, fg_pixels
    # for debug - better view:
    # bg_pixels = np.transpose((np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)).nonzero())
    # fgPixels = np.transpose((np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)).nonzero())
    # print(bg_pixels, fg_pixels)


def get_pixels_for_train(img, bg_pixels, fg_pixels):
    return img[bg_pixels], img[fg_pixels]


def create_GMM(pixels_for_train):
    global n_comp
    return GMM(n_comp, covariance_type='full', init_params='kmeans', random_state=0).fit(pixels_for_train)


def initalize_GMMs(img, mask, n_components=5):
    # TODO: implement initalize_GMMs --> check if GMM default function is okay
    global n_comp
    n_comp = n_components
    bg_pixels, fg_pixels = split_bg_fg_pixels(mask)
    bg_pixels_for_train, fg_pixels_for_train = get_pixels_for_train(img, bg_pixels, fg_pixels)
    bgGMM = create_GMM(bg_pixels_for_train)
    fgGMM = create_GMM(fg_pixels_for_train)
    return bgGMM, fgGMM


def update_GMM_weights(gmm, unique_labels, count):
    global n_comp
    new_weights = np.zeros(n_comp)
    num_of_samples = np.sum(count)
    for i, label in enumerate(unique_labels):
        new_weights[label] = count[i]/num_of_samples
    gmm.weights_ = new_weights


def update_GMM_means(gmm, n_features, pixels, labels, unique_labels):
    global n_comp
    new_means = np.zeros((n_comp, n_features))
    for label in unique_labels:
        new_means[label] = np.mean(pixels[label == labels], axis=0)
    gmm.means_ = new_means


def update_GMM_covariance_matrix(gmm, n_features, pixels, labels, unique_labels, count):
    global n_comp
    new_covariance_matrix = np.zeros((n_comp, n_features, n_features))
    for i, label in enumerate(unique_labels):
        if count[i] <= 1:
            new_covariance_matrix[label] = 0
        else:
            new_covariance_matrix[label] = np.cov(np.transpose(pixels[label == labels]))
    gmm.covariances_ = new_covariance_matrix


def update_GMM_fields(pixels, gmm):
    n_features = gmm.n_features_in_
    labels = gmm.predict(pixels)
    unique_labels, count = np.unique(labels, return_counts=True)
    # Update weights, means, covariance_matrix
    update_GMM_weights(gmm, unique_labels, count)
    update_GMM_means(gmm, n_features, pixels, labels, unique_labels)
    update_GMM_covariance_matrix(gmm, n_features, pixels, labels, unique_labels, count)


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bg_pixels, fg_pixels = split_bg_fg_pixels(mask)
    bg_pixels_for_train, fg_pixels_for_train = get_pixels_for_train(img, bg_pixels, fg_pixels)
    update_GMM_fields(bg_pixels_for_train, bgGMM)
    update_GMM_fields(fg_pixels_for_train, fgGMM)
    return bgGMM, fgGMM


def add_n_edges(edges_n_link, weights_n, indices_img1, indices_img2, weight):
    slice_img_1 = indices_img1.flatten()
    slice_img_2 = indices_img2.flatten()
    edges_n_link.extend(list(zip(slice_img_1, slice_img_2)))
    weights_n.extend(list(weight.flatten()))
    return weight.flatten()


# N-link - 4 edges for 4 neighbors: left, up, upleft, upright
# “n-links” represent local information about a pixel and its direct surroundings
def calculate_n_links():
    global rows, columns, weight_left, weight_up, weight_upleft, weight_upright
    edges_n_link = []
    weights_n = []
    indices_img = np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns)

    # Left neighbor
    add_n_edges(edges_n_link, weights_n, indices_img[:, 1:], indices_img[:, :-1], weight_left)
    # Up neighbor
    add_n_edges(edges_n_link, weights_n, indices_img[1:, :], indices_img[:-1, :], weight_up)
    # Upleft neighbor
    add_n_edges(edges_n_link, weights_n, indices_img[1:, 1:], indices_img[:-1, :-1], weight_upleft)
    # Upright neighbor
    add_n_edges(edges_n_link, weights_n, indices_img[1:, :-1], indices_img[:-1, 1:], weight_upright)

    return edges_n_link, weights_n


# Calculate the probability each sample belongs to specific component in GMM
# According to formula (2) in "Implementing GrabCut" document
def calculate_probability_for_component(samples, component, gmm):
    res = np.zeros(samples.shape[0])
    if gmm.weights_[component] > 0:
        sub = samples - gmm.means_[component]
        sub_t = np.transpose(sub)
        power = np.sum(sub * np.transpose(np.dot(np.linalg.inv(gmm.covariances_[component]), sub_t)), axis=1)
        # TODO: in formula 9 in original and formula 2 in document 0.5 is without minus
        res = np.exp(-0.5 * power) / np.sqrt(2 * np.pi) / np.sqrt(np.linalg.det(gmm.covariances_[component]))
    return res


# For each sample return the probability to belong each component in GMM
def calculate_probabilities(samples, gmm):
    global n_comp
    return np.array([calculate_probability_for_component(samples, c, gmm) for c in range(n_comp)])


# Return the most likely component in GMM to each sample
def GMM_component(samples, gmm):
    return np.argmax(np.transpose(calculate_probabilities(samples, gmm)), axis=1)


# Calculate the probability each sample belongs to the GMM
def calculate_probability_for_GMM(samples, gmm):
    return np.dot(gmm.weights_, calculate_probabilities(samples, gmm))


def assign_GMM_components_to_pixels(img, bgGMM, fgGMM, bg_pixels, fg_pixels):
    global rows, columns
    pixels_components = np.zeros(rows, columns)
    pixels_components[bg_pixels] = GMM_component(img[bg_pixels], bgGMM)
    pixels_components[fg_pixels] = GMM_component(img[fg_pixels], fgGMM)
    return pixels_components


# T-link
# “t-links” represent global information about color distribution in the foreground and the background of the image.
# A t-link weight shows how well a pixel fits the background/foreground model.
# There are two T-links for each pixel:
# 1.The Background T-link connects the pixel to the Background node.
# 2.The Foreground T-link connects the pixel to the Foreground node.
def calculate_t_links(img, mask, bgGMM, fgGMM):
    global rows, columns, k
    edges_t_link = []
    weights_t = []

    flatten_mask = mask.flatten()
    bg_pixels = (flatten_mask == GC_BGD).nonzero()
    fg_pixels = (flatten_mask == GC_FGD).nonzero()
    pr_pixels = ((flatten_mask == GC_PR_BGD) | (flatten_mask == GC_PR_FGD)).nonzero()

    grid = rows * columns
    foreground_node = grid
    background_node = grid + 1

    # According to formula (2) in "Implementing GrabCut" document
    # Pr_pixels
    edges_t_link.extend(list(zip([foreground_node] * pr_pixels[0].size, pr_pixels[0])))
    D = -np.log(calculate_probability_for_GMM(np.reshape(img, (grid, 3))[pr_pixels], bgGMM))
    weights_t.extend(D.tolist())

    edges_t_link.extend(list(zip([background_node] * pr_pixels[0].size, pr_pixels[0])))
    D = -np.log(calculate_probability_for_GMM(np.reshape(img, (grid, 3))[pr_pixels], fgGMM))
    weights_t.extend(D.tolist())

    # Bg_pixels
    edges_t_link.extend(list(zip([foreground_node] * bg_pixels[0].size, bg_pixels[0])))
    weights_t.extend([0]*bg_pixels[0].size)

    edges_t_link.extend(list(zip([background_node] * bg_pixels[0].size, bg_pixels[0])))
    weights_t.extend([k]*bg_pixels[0].size)

    # Fg_pixels
    edges_t_link.extend(list(zip([foreground_node] * fg_pixels[0].size, fg_pixels[0])))
    D = -np.log(calculate_probability_for_GMM(np.reshape(img, (grid, 3))[fg_pixels], bgGMM))
    weights_t.extend([k]*fg_pixels[0].size)

    edges_t_link.extend(list(zip([background_node] * fg_pixels[0].size, fg_pixels[0])))
    weights_t.extend([0]*fg_pixels[0].size)

    return edges_t_link, weights_t


# According to the document "Implementing GrabCut" k is a large constant value
# calculated as follows to ensure that it is the largest weight in the graph:
# k = max_m ∑_(n:(m,n)εE) N(m,n)
# k<= max_((m,n)εE) N(m,n)*8 (Num of neighbors smaller than 8)
def calculate_k(weights_n):
    global k
    if not k:
        k = 8 * np.max(weights_n)


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    global rows, columns, k
    min_cut = [[], []]
    energy = 0

    edges_n_link, weights_n = calculate_n_links()
    calculate_k(weights_n)
    edges_t_link, weights_t = calculate_t_links(img, mask, bgGMM, fgGMM)

    graph = igraph.Graph(columns * rows + 2)
    graph.add_edges(edges_n_link)
    graph.add_edges(edges_t_link)

    weights = weights_n + weights_t

    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
