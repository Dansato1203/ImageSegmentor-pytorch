import torch
import numpy as np

def load_image(fname, imw, imh):
	img = Image.open(fname).resize((imw, imh))
	a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.

	return a, img

# save torchscript for C++
a, img = load_image('/home/citbrains/Dan/100_test/000020.jpg', 320, 240)
#a1 = np.expand_dims(a,axis=0)
ex = np.expand_dims(a, axis=0)
traced_script_module = torch.jit.trace(model, torch.FloatTensor(ex))
traced_script_module.save(weight_name) # これをC++から呼び出す
