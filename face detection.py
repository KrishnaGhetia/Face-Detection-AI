import cv2 as cv
from google.colab.patches import cv2_imshow

img = cv.imread('imgaa.jpeg')
cv2_imshow(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv2_imshow(gray)

haar_cascade = cv.CascadeClassifier('face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv2_imshow(img)



cv.waitKey(0)