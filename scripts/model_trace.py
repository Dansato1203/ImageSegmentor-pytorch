import torch
import MLP
import numpy as np
import argparse
from PIL import Image

def load_image(fname, imw, imh):
	img = Image.open(fname).resize((imw, imh))
	a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
	return a, img

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, default='traced_sf_model.pt', 
                    help='Name of weight to save')
args = parser.parse_args()

model_path = "train_result/sf_model.pt"

model = MLP.MLP(4, 3)
model.load_state_dict(torch.load(model_path))
model.eval()

# save torchscript for C++
a, img = load_image('/train/test_dataset/000317.jpg', 320, 240)
#a1 = np.expand_dims(a,axis=0)
ex = np.expand_dims(a, axis=0)
traced_script_module = torch.jit.trace(model, torch.FloatTensor(ex))
traced_script_module.save("train_result/" + args.weight) # これをC++から呼び出す
