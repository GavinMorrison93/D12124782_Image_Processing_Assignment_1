# To accomplish this task I did the following:

# Stage 1. I converted the image to grayscale.

# Stage 2. I then applied Contrast Limited Adaptive Histogram  
# Equalization to try and correct the uneven lighting on the page.

# Stage 3. I then applied an Adaptive Threshold which turned 
# the information on the page black or white and therefore 
# getting rid of a lot of detail making the text unreadable. 
# So I ended up using the threshold just to assist canny find 
# the edges of the ROI. 

# Stage 4. I used the opencv'canny' function to find the edges of my
# ROI (Region of Interest)

# Stage 5. I used the numpy 'argwhere' function to crop the image
# using the edges located by the 'canny' function.

# Final stage. I output a copy of the corrected image and display 
# the corrected image on screen.

# Summary: I was hoping that my result would be a crisp white 
# background with clear black text in the foreground, but as the 
# priority was to make the text easier to read I had to make a 
# call with that in mind, so in terms of contrast, I think my 
# program has provided results somewhere in the middle, but 
# the text is definitely more readable. 

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

stage1 = cv2.imread("text.jpg", cv2.IMREAD_GRAYSCALE)

stage2a = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
stage2b = stage2a.apply(stage1)

stage3 = cv2.adaptiveThreshold(stage2b, maxValue = 255,
  adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
  thresholdType = cv2.THRESH_BINARY,
  blockSize = 5, C = 15)

height = np.size(stage3, 0)
width = np.size(stage3, 1)

stage4 = cv2.Canny(stage3, height, width)

stage5a = stage4 > 0

stage5b = np.argwhere(stage5a)

horiz0, vertic0 = stage5b.min(axis = 0)
horiz1, vertic1 = stage5b.max(axis = 0) + 1

stage5d = stage2b[horiz0: horiz1, vertic0: vertic1]

cv2.imwrite('Corrected_image.jpg', stage5d)

cv2.imshow("Corrected Image", stage5d)
key = cv2.waitKey(0)