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

    # transpose Xkernel to find y derivative
    Ykernel = np.transpose([Xkernel])

    w = win_size // 2

    # normalize pixels
    I1g = im1 / 255.

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

    print(np.array(originalPoints)[0][0])
    print(np.array(dU_dV)[0])
    return np.array(originalPoints), np.array(dU_dV)

def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False

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

    # if RGB make grayscale
    if (not isgray(img1)):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if (not isgray(img2)):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find gaussian pyramids
    gaussPyr1 = gaussianPyr(img1, k)
    gaussPyr2 = gaussianPyr(img2, k)

    # create answer array
    height, width = img1.shape
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    ans_shape = (height, width, 2)
    ans = np.zeros(ans_shape)

    # find optical flow for every layer and add
    for i in range(k - 1, 1, -1):
        pts, UV = opticalFlow(gaussPyr1[i], gaussPyr2[i], stepSize, winSize)
        for j in range(pts.shape[0]):
            x = pts[j][0]
            y = pts[j][1]
            u[x][y] += UV[j][0]
            v[x][y] += UV[j][1]
        u = u * 2
        v = v * 2

    pts, UV = opticalFlow(gaussPyr1[0], gaussPyr2[0], stepSize, winSize)
    for j in range(pts.shape[0]):
        if(pts[j][0]<=400):
            x = pts[j][0]
        y = pts[j][1]
        u[x][y] += UV[j][0]
        v[x][y] += UV[j][1]
    u = u * 2
    v = v * 2

    # move into answer array
    for i in range(height):
        for j in range(width):
            ans[i][j][0] = u[i][j]
            ans[i][j][1] = v[i][j]

    return ans


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """

    # create matrix
    t = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    good_features = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')

    # find optical flow
    cv_lk_pyr = cv2.calcOpticalFlowPyrLK(im1, im2, good_features, None)[0]
    directions = cv_lk_pyr - good_features
    curr_mse = np.inf

    # go over all direction in for loop from lk (built in one?)
    # add u and v to unit matrix
    for i in range(len(directions)):
        u = directions[i, 0, 0]
        v = directions[i, 0, 1]
        t[0][2] = u
        t[1][2] = v
        # warp img 1, t,im1.shape[::-1]
        img_2 = cv2.warpPerspective(im1, t, im1.shape[::-1])
        mse = np.square(im1 - img_2).mean()
        if mse < curr_mse:
            best = img_2
            curr_mse = mse

    # calculate mse
    mse = np.square(im1-best).mean()

    print("MSE: ",mse)

    return best


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    good_features = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')

    # find optical flow
    cv_lk_pyr = cv2.calcOpticalFlowPyrLK(im1, im2, good_features, None)[0]
    directions = cv_lk_pyr - good_features
    curr_mse = np.inf

    # go over directions in u,v
    for i in range(len(directions)):
        u = directions[i, 0, 0]
        v = directions[i, 0, 1]
        # find angle
        if u==0:
            angle = 0
        else:
            angle = np.arctan(v/u)
        # create matrix
        matrix = np.array([[np.cos(angle), -np.sin(angle),0],
                           [np.sin(angle),np.cos(angle),0],
                          [0,0,1]],dtype = float)
        # warp
        img_2 = cv2.warpPerspective(im1, matrix, im1.shape[::-1])
        # calculate mse, keep image that gives best result
        mse = np.square(im1 - img_2).mean()
        if mse < curr_mse:
            best = img_2
            curr_mse = mse
    # find lk translation for image we kept and img2
    translated = findTranslationLK(best,im2)

    # the answer is the translation @ rotation
    answer = translated @ img_2

    return answer


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    return im1


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    return im1


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    # find T inverse
    inverse = np.linalg.inv(T)

    # create answer array
    answer = np.zeros_like(im1)

    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            a = np.array([i, j, 1])
            b = inverse.dot(a)
            x = int(b[0])
            y = int(b[1])
            if 0<=x<im1.shape[0] and 0<=y<im1.shape[1]:
                answer[i][j]=im1[x][y]

    # calculate mse
    mse = np.square(im1 - answer).mean()

    print("MSE: ",mse)

    return answer


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

    # create answer array
    answer = []

    # add 'smallest' image to answer
    answer.append(img)
    blurred = img

    # add all images to answer array after blurring
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
    # create answer array
    answer = []

    # find gaussian pyramid
    img_list = gaussianPyr(img,levels)

    # add images to answer
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
    # add last image to answer
    answer = lap_pyr[-1]

    # add images to answer
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
    # find blended image
    lapA = laplaceianReduce(img_1,levels)
    lapB = laplaceianReduce(img_2,levels)
    gaussPyr = gaussianPyr(mask,levels)

    # create answer array
    lapC = []

    # for every level in the pyramids
    for k in range(levels):
        img = gaussPyr[k]*lapA[k]+(1-gaussPyr[k])*lapB[k]
        lapC.append(img)

    # Reconstruct all levels to get blended image
    blended = laplaceianExpand(lapC)
    blended = np.resize(blended,[679, 1023, 3])

    # find naive blend
    naive = img_1*mask+img_2*(1-mask)

    return np.array(naive), np.array(blended)

