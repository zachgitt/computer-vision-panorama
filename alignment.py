import math
import random

import cv2
import numpy as np
import pdb

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        A[i*2, :] = [float(a_x), float(a_y), 1., 0., 0., 0.,float(-b_x*a_x), float(-b_x*a_y), float(-b_x)]
        A[i*2+1, :] = [0., 0., 0., float(a_x), float(a_y), 1., float(-b_y*a_x), float(-b_y*a_y), float(-b_y)]

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = (Vt[-1]/Vt[-1][8]).reshape(3, 3)
    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''
    # N trials
    max_inliers = []
    for n in range(nRANSAC):
        # Current motion model
        inliers = []
        # Translation
        if m == eTranslate:
            match = matches[random.randint(0, len(matches)-1)]
            f1_pt = f1[match.queryIdx].pt
            f2_pt = f2[match.trainIdx].pt
            tx = f2_pt[0] - f1_pt[0]
            ty = f2_pt[1] - f1_pt[1]
            M = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            inliers = getInliers(f1, f2, matches, M, RANSACthresh)
        # Homography
        elif m == eHomography:
            h_matches = np.random.choice(matches, size=4, replace=False)
            M = computeHomography(f1, f2, h_matches)
            inliers = getInliers(f1, f2, matches, M, RANSACthresh)
        # Best motion model
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
    return leastSquaresFit(f1, f2, matches, m, max_inliers)


def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        # Get tuple points
        pt1 = f1[matches[i].queryIdx].pt
        pt2 = f2[matches[i].trainIdx].pt

        # Convert to lists
        pt1 = np.array([pt1[0], pt1[1], 1])
        pt2 = np.array([pt2[0], pt2[1], 1])
        # Predict
        pt2_pred = np.dot(M, pt1)
        pt2_pred[0] = pt2_pred[0]/pt2_pred[2]
        pt2_pred[1] = pt2_pred[1]/pt2_pred[2]
        if np.linalg.norm(pt2_pred - pt2) < RANSACthresh:
            inlier_indices.append(i)
    return inlier_indices



def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            pt1 = f1[matches[inlier_indices[i]].queryIdx].pt
            pt2 = f2[matches[inlier_indices[i]].trainIdx].pt
            u += pt2[0] - pt1[0]
            v += pt2[1] - pt1[1]

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        M = computeHomography(f1, f2, [matches[i] for i in inlier_indices])

    else:
        raise Exception("Error: Invalid motion model.")
    return M



