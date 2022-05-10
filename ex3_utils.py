import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206664278

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # kernel for finding x derivative
    Xkernel = np.array([[1, 0, -1]])
    # transpose Xkernel for finding y derivative
    Ykernel = np.transpose([Xkernel])

    w = win_size // 2
    # normalize pixels
    I1g = im1 / 255.
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    fx = cv2.filter2D(im2, -1, Xkernel, borderType=cv2.BORDER_REPLICATE)
    fy = cv2.filter2D(im2, -1, Ykernel, borderType=cv2.BORDER_REPLICATE)
    ft = im2-im1

    originalPoints = []
    dU_dV = []
    for i in range(w, I1g.shape[0] - w,step_size):
        for j in range(w, I1g.shape[1] - w,step_size):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            AtA_ = [[(Ix * Ix).sum(), (Ix * Iy).sum()],
                    [(Ix * Iy).sum(), (Iy * Iy).sum()]]
            lam_ = np.linalg.eigvals(AtA_)
            lam2_ = np.min(lam_)
            lam1_ = np.max(lam_)
            # the checks for λ1/2:
            if lam2_ <= 1 or lam1_ / lam2_ >= 100:
                continue
            Atb_ = [[-(Ix * It).sum()], [-(Iy * It).sum()]]
            v_ = np.linalg.inv(AtA_) @ Atb_
            dU_dV.append(v_)
            originalPoints.append([j, i])

    return np.array(originalPoints), np.array(dU_dV)



def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    n = 2 ** levels
    width, height = n * int(img.shape[1] / n), n * int(img.shape[0] / n)
    img = cv2.resize(img, (width, height))
    answer = []
    answer.append(img)
    blurred = img
    for i in range(levels-1):
        blurred = cv2.blur(blurred,(5,5))
        blurred = blurred[::2, ::2]
        answer.append(blurred)
    return answer


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    answer = []
    img_list = gaussianPyr(img,levels)
    for i in range(levels):
        curr=img_list[i]
        answer.append(curr-cv2.blur(curr,(5,5)))
    return answer


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    answer = lap_pyr[-1]
    for img in range(len(lap_pyr)-1,0,-1):
        answer=cv2.pyrUp(answer)
        answer=answer+lap_pyr[img-1]
    return answer


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass

