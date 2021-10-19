#! usr/bin/env python3

from PIL import Image
import numpy as np

origin_img = Image.open('../test_images/000650.jpg')
w_label = Image.open('../test_labels/white_label/000650_label.png')
g_label = Image.open('../test_labels/green_label/000650_label.png')

print("g: ", g_label.mode)
print("w: ", w_label.mode)
print("o: ", origin_img.mode)

palette = g_label.getpalette()
print(f"shape: {len(palette)}")
g_lamel.show()

w_label = w_label.convert('P')
w_label.show()


print(g_label.getpixel((300, 300)))

empty_img = Image.new('P', (640, 480))

for y in range(480):
  for x in range(640):
    g = g_label.getpixel((x, y))
    w = w_label.getpixel((x, y))

    empty_img.putpixel((x, y), g)
    if w != 0:
      empty_img.putpixel((x, y), 50)

empty_img.putpalette(palette)
empty_img.show()

"""
origin_img.show()
empty_img.show()
"""
