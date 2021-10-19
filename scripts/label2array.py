#! usr/bin/env python3

import numpy as np
import cv2
from PIL import Image
import random

imw = 640
imh = 480

def load_img(fname, imw, imh):
  img = Image.open(fname).resize((imw, imh))

  a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
  return a, img

def png2array(lfile_array, imw, imh):
  a = np.zeros((3, imh, imw), dtype=np.float32)
  for x in range(imw):
    for y in range(imh):
      if lfile_array[y][x] == 1:
        a[0][y][x] = 1
      elif lfile_array[y][x] == 50:
        a[1][y][x] = 1
      else:
        a[2][y][x] = 1
  return a

x = np.zeros((0, 3, imh, imw), dtype=np.float32)
t = np.zeros((0, 3, imh, imw), dtype=np.float32)

f = '../test_images/000435.jpg'

print(x.shape)
a, img = load_img(f, imw, imh)
print(a.shape)
img.show()
a1 = np.expand_dims(a, axis=0)
print(a1.shape)
x = np.append(x, a1, axis=0)
print(x.shape)

lfile = '../test_labels/composition_label/000435_label.png'
print(lfile)
#img = Image.open(lfile)
img = cv2.imread(lfile, cv2.IMREAD_GRAYSCALE)

a = np.asarray(img).astype(np.float32)
a = png2array(a,imw,imh)
print(a.shape)
a1 = np.expand_dims(a, axis=0)
print(a1.shape)
t = np.append(t, a1, axis=0)
print(t.shape)
print(t)

cv2.imshow("image",img)
cv2.waitKey(0)
#img.show()
