import cv2
import numpy as np

# 讀取圖片
img = cv2.imread("videoframe_189261.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 偵測圓形
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=100, param2=30, minRadius=20, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i, (x, y, r) in enumerate(circles[0, :]):
        # 擷取圓形區域
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        out = cv2.bitwise_and(img, img, mask=mask)

        # 裁切 bounding box
        cropped = out[y-r:y+r, x-r:x+r]
        cv2.imwrite(f"circle_{i}.png", cropped)