import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    height, width, _ = img.shape
    pts_in = np.array([
        [0, 0, 1],
        [0, width-1, 1],
        [height-1, 0, 1],
        [height-1, width-1, 1]
    ])
    # Calculate transforms
    pts_out = np.dot(M, pts_in.T)
    pts_out[0] = (pts_out[0]/pts_out[2])
    pts_out[1] = (pts_out[1]/pts_out[2])
    minX = np.min(pts_out[0])
    maxX = np.max(pts_out[0])
    minY = np.min(pts_out[1])
    maxY = np.max(pts_out[1])
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # Get warped image
    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    warped = cv2.warpPerspective(img, M, (acc.shape[1], acc.shape[0]), flags=1)
    for y in range(minY, maxY):
        for x in range(minX, maxX):
            #Check if y and x are in bounds, if not continue
            if not is_inbounds(x, y, acc):
                continue
            if warped[y][x][0] == 0 and warped[y][x][1] == 0 and warped[y][x][2] == 0:
                continue
            if x - minX < blendWidth:
                weight = (x-minX)/blendWidth
                for color in range(3):
                    acc[y][x][color] += weight * warped[y][x][color]
                acc[y][x][3] += weight
            elif maxX - x < blendWidth:
                weight = (maxX-x)/blendWidth
                for color in range(3):
                    acc[y][x][color] += weight * warped[y][x][color]
                acc[y][x][3] += weight
            else:
                for color in range(3):
                    acc[y][x][color] += warped[y][x][color]
                acc[y][x][3] += 1
    return acc




def is_inbounds(x, y, img):
    return y < img.shape[0] and x < img.shape[1] and x>=0 and y>=0

def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    height, width, depth = acc.shape
    final_img = np.zeros((height, width, depth-1)).astype('uint8')
    for i in range(height):
        for j in range(width):
            for k in range(3):
                if acc[i, j, 3] != 0 and (acc[i, j, k]/acc[i, j, 3]) > 255:
                    acc[i, j, k] = 255
                elif acc[i, j, 3] != 0:
                    final_img[i, j, k] = acc[i, j, k] / acc[i, j, 3]
    return final_img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        height, width, _ = img.shape
        pts_in = np.array([
            [0, 0, 1],
            [0, width - 1, 1],
            [height - 1, 0, 1],
            [height - 1, width - 1, 1]
        ])

        # Calculate transforms
        pts_out = np.dot(M, pts_in.T)
        minX = np.min(np.append(pts_out[0], minX))
        maxX = np.max(np.append(pts_out[0], maxX))
        minY = np.min(np.append(pts_out[1], minY))
        maxY = np.max(np.append(pts_out[1], maxY))

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates

    if(is360):
        A = computeDrift(x_init, y_init, x_final, y_final, width)


    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

