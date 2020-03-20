from skimage import data
from PIL import Image
import numpy as np
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


jpg = Image.open(r"C:\Users\arno\Documents\GitHub\scikit-image-orb-descriptor-dual-image-mapping-d\day.jpg")
jpg1 = Image.open(r"C:\Users\arno\Documents\GitHub\scikit-image-orb-descriptor-dual-image-mapping-d\night1.jpg")

MatImg=np.array(jpg)
MatImg1=np.array(jpg1)
img1 = rgb2gray(MatImg)
img2 = rgb2gray(MatImg1)

keypoints1 = corner_peaks(corner_harris(img1), min_distance=5)
keypoints2 = corner_peaks(corner_harris(img2), min_distance=5)

extractor = BRIEF()

extractor.extract(img1, keypoints1)
keypoints1 = keypoints1[extractor.mask]
descriptors1 = extractor.descriptors

extractor.extract(img2, keypoints2)
keypoints2 = keypoints2[extractor.mask]
descriptors2 = extractor.descriptors


matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

fig, ax = plt.subplots()

plt.gray()

plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
ax.axis('off')
ax.set_title("Original Image vs. Transformed Image : BRIEF")


plt.show()