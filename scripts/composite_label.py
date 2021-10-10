#! usr/bin/env python3

from PIL import Image
import numpy as np
import glob
import os
import argparse
import cv2

imw = 640
imh = 480

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label_dir")
    args = parser.parse_args()

    label_dir = args.label_dir
    
    w_images = glob.glob('../' + label_dir + '/white_label/*.png')
    g_images = glob.glob('../' + label_dir + '/green_label/*.png')

    composite_labels = '../' + label_dir + '/composition_label'
    if not os.path.exists(composite_labels):
      os.mkdir(composite_labels)

    if len(w_images) == len(g_images):
        print(f"{len(w_images)} files for composition")
        for i in range(len(w_images)):
            print(f"{i+1}  :",end='')
            w_label = Image.open(w_images[i]).resize((imw, imh))
            g_label = Image.open(g_images[i]).resize((imw, imh))

            #w_label.show()
            #g_label.show()
      
            if w_label.mode != 'P':
              w_label = w_label.convert('P')
            if g_label.mode != 'P':
              g_label = g_label.convert('P')

            #w_label = np.asarray(w_label).astype(np.int32)
            #g_label = np.asarray(g_label).astype(np.int32)

            palette = g_label.getpalette()
            
            #h, w = np.asarray(w_label).shape
            #w, h = w_label.size

            empty_img = Image.new('P', (imw, imh))
            #img_gray = cv2.imread(lfile, cv2.IMREAD_GRAYSCALE)
            #cv2.imshow(img_gray)

            for y in range(imh):
              for x in range(imw):
                #r1, g1, b1 = g_label.getpixel((x, y))
                #r2, g2, b2 = w_label.getpixel((x, y))

                #g = g_label[y][x]
                #w = w_label[y][x]
                
                g = g_label.getpixel((x, y))
                w = w_label.getpixel((x, y))

                #print(f"g:[ {g} ]")
                #print(f"w:[ {w} ]")

                empty_img.putpixel((x, y), g)
                #empty_img[y][x] = g
                if w != 0:
                  empty_img.putpixel((x, y), 50)
                    #empty_img[y][x] = w

            #empty_img = Image.fromarray(empty_img, mode='P')
            empty_img.putpalette(palette)
            #empty_img.show()

            folda_name = os.path.split(w_images[i])
            file_name = os.path.splitext(folda_name[-1])
            empty_img.save(os.path.join(composite_labels, file_name[0] + ".png"))
            print(" Done!")
    else:
        print("dekinaiyo")
    

if __name__ == "__main__":
    main()
