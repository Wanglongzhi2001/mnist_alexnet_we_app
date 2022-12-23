import cv2

img_wlz = cv2.imread('./static/lc.jpg')

img_wlz = cv2.resize(img_wlz, (64, 64))
cv2.imwrite("result_lc.jpg", img_wlz)