import cv2
import numpy as np

W = 1920 // 2
H = 1080 // 2

cap = cv2.VideoCapture('test_countryroad.mp4')

orb = cv2.ORB_create()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (W, H))
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, (0, 255, 0))
    # kp, des = orb.compute(gray, kp)
    # img_m = cv2.drawKeypoints(frame, kp, None, (0, 255, 0), flags=0)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
