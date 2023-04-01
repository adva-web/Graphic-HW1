import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

#############
# CONSTANTS #
#############
GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

####################
# GLOBAL VARIABLES #
####################


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    # print(list(mask))
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD
    print(mask[y:y+h, x:x+w])
    print(mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2])

    bgGMM, fgGMM = initalize_GMMs(img, mask)

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



# returns the background pixels and foreground pixels of image according to mask
def split_bg_fg_pixels(mask):
    bgPixels = ((mask == GC_BGD) | (mask == GC_PR_BGD)).nonzero()
    fgPixels = ((mask == GC_FGD) | (mask == GC_PR_FGD)).nonzero()
    # for debug - better view:
    # bgPixels = np.transpose((np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)).nonzero())
    # fgPixels = np.transpose((np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)).nonzero())
    # print(bgPixels, fgPixels)
    return bgPixels, fgPixels


def getPixelsForTrain(img, bgPixels, fgPixels):
    return img[bgPixels], img[fgPixels]


# def initGMM(img, pixels):
#     pixelsForTrain = img[pixels]
#     return None

def init_GMM(n_components, bgPixelsForTrain, fgPixelsForTrain):
    bgGMM = GMM(n_components, covariance_type='full', init_params='kmeans', random_state=0).fit(bgPixelsForTrain)
    fgGMM = GMM(n_components, covariance_type='full', init_params='kmeans', random_state=0).fit(fgPixelsForTrain)
    return bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    # TODO: implement initalize_GMMs --> check if GMM default function is okay
    bgPixels, fgPixels = split_bg_fg_pixels(mask)
    # bgGMM = initGMM(img, bgPixels)
    # fgGMM = initGMM(img, bgPixels)
    bgPixelsForTrain, fgPixelsForTrain = getPixelsForTrain(img, bgPixels, fgPixels)
    return init_GMM(n_components, bgPixelsForTrain, fgPixelsForTrain)


def update_GMM_weights(pixels, gmm):
    n_components = len(gmm.weights_)
    new_weights = np.zeros(n_components)
    labels = gmm.predict(pixels)
    unique_lables, count = np.unique(labels, return_counts=True)
    num_of_samples = np.sum(count)
    for i, label in enumerate(unique_lables):
        new_weights[label] = count[i]/num_of_samples
    gmm.weights_ = new_weights


def update_GMM_means(pixels, gmm):
    n_components = gmm.means_.shape[0]
    n_features = gmm.means_.shape[1]
    new_means = np.zeros((n_components, n_features))
    labels = gmm.predict(pixels)
    unique_lables, count = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique_lables):
        new_means[label] = np.mean(pixels[label == labels], axis=0)
    gmm.means_ = new_means


def update_GMM_fields(pixels, gmm):
    # update_GMM_weights(pixels, gmm)
    update_GMM_means(pixels, gmm)
    # update_GMM_covariance_matrix(pixels, gmm)


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    bgPixels, fgPixels = split_bg_fg_pixels(mask)
    bgPixelsForTrain, fgPixelsForTrain = getPixelsForTrain(img, bgPixels, fgPixels)
    update_GMM_fields(bgPixelsForTrain, bgGMM)
    update_GMM_fields(fgPixelsForTrain, fgGMM)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0
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
