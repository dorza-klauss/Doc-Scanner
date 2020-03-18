import numpy as np
import cv2

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] #top left corner
    rect[2] = pts[np.argmax(s)] #bottom right corner

    diff = pts.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] #top right corner
    rect[3] = pts[np.argmax(diff)]#bottom left corner

    return rect

def four_point_transform(image,pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    #computing the width for the new perspective
    WidthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    WidthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(WidthA), int(WidthB))

    #computing the height for the new perspective

    HeightA = np.sqrt(((br[0]-tr[0])**2)+((br[1]-tr[1])**2))
    HeightB = np.sqrt(((bl[0]-tl[0])**2)+((bl[1]-tl[1])**2))
    maxHeight = max(int(HeightA), int(HeightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped




