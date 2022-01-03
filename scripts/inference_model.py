import torch
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
 

import MLP

def load_image(fname, imw, imh):
	img = Image.open(fname).resize((imw, imh))

	a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
	return a, img

def eval_image(fname, thre):
	imw = 320
	imh = 240
	testx = np.zeros((0, 3, imh, imw), dtype=np.float32)

	a, img = load_image(fname, imw, imh)
	a1 = np.expand_dims(a,axis=0)

	testx = np.append(testx, a1, axis=0)

	t0 = time.time()
	testy = model(torch.FloatTensor(testx))
	print(testy.shape)
	print('forward time [s]: ' + str(time.time()-t0))

	imd = Image.new('RGB', (imw*3, imh))
    
	thimg_g = (testy.to(cpu).detach().numpy()[0][0] > thre) * 255
	thimg_w = (testy.to(cpu).detach().numpy()[0][1] > thre) * 255

	print(f'max {np.max(thimg_g)}')
	print(f'min {np.min(thimg_g)}')

	print(f'max {np.max(thimg_w)}')
	print(f'min {np.min(thimg_w)}')

	thimg_g = thimg_g.astype(np.uint8)
	thimg_w = thimg_w.astype(np.uint8)

	thimg_g = Image.fromarray(thimg_g).convert('L')
	thimg_w = Image.fromarray(thimg_w).convert('L')

	imd.paste(img, (0,0))
	imd.paste(thimg_g, (imw, 0))
	imd.paste(thimg_w, (imw*2, 0))
	plt.figure(figsize=(16,9))
	plt.imshow(imd)

cpu = torch.device("cpu")

model_path = "wl_model.pt"

model = MLP.MLP(4, 3)
model.load_state_dict(torch.load(model_path))
model.eval()

eval_image('/home/citbrains/Dan/100_test/000020.jpg', 0.5)
eval_image('/home/citbrains/Dan/100_test/000143.jpg', 0.5)

