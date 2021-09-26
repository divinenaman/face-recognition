from cv2 import imread
from cv2 import CascadeClassifier
import os

pixels = imread(os.path.join('assets','test1.jpg'))

classifier = CascadeClassifier(os.path.join('pretrained-models','haarcascade_frontalface_default.xml'))

bboxes = classifier.detectMultiScale(pixels)

for box in bboxes:
	print(box)