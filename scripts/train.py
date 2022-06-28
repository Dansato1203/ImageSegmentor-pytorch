import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import os
import time

import MLP

IMW = 320
IMH = 240

def load_image(fname, imw, imh):
	img = Image.open(fname).resize((imw, imh))

	a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
	return a, img

def png2array(lfile_array, imw, imh):
	a = np.zeros((imh, imw), dtype=np.float32)
	for x in range(imw):
		for y in range(imh):
			if (lfile_array[y][x] == 40):
				a[y][x] = 0
			elif (lfile_array[y][x] == 225):
				a[y][x] = 1
			else:
				a[y][x] = 2

	return a

class SoccerFieldDataset(Dataset):
	def __init__(self, image_dir, transform=None):
		imw = IMW
		imh = IMH
		x = np.zeros((0, 3, imh, imw), dtype=np.float32)
		t = np.zeros((0, imh, imw), dtype=np.float32)
		jpgfiles = glob.glob(image_dir + '*.jpg')
		print(jpgfiles)
		for f in jpgfiles:
			a, img = load_image(f, imw, imh)
			a1 = np.expand_dims(a,axis=0)
			x = np.append(x, a1, axis=0)
            
			lfile = os.path.splitext(f)[0] + '_label.png'
			print(lfile)
			img = Image.open(lfile).resize((imw, imh))
            
			a = np.asarray(img).astype(np.float32)
			print(f"a: {a.shape}")
			#a = a.astype(np.int32)
			a = png2array(a, imw, imh)
			print(f"a: {a.shape}")
			a1 = np.expand_dims(a,axis=0)
			t = np.append(t, a1, axis=0)
			print(f"t: {t.shape}")
        
		self.data  = x
		self.label = t

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		images = self.data[idx, :, :, :]
		labels = self.label[idx, :, :]

		return (images, labels)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args(args=[])

dataset = SoccerFieldDataset('train_dataset/')
#val_dataset = SoccerFieldDataset('val_dataset/')
train_loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=0)
#val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=0)

cpu = torch.device("cpu")

# exec training
def train(args, model, device, dataloader, optimizer, epoch):
	model.train()
	lossfun = nn.CrossEntropyLoss()
	for batch_idx, (train_data, train_target) in enumerate(dataloader):
		train_data, train_target = train_data.to(device), train_target.to(device, dtype=torch.long)
		optimizer.zero_grad()
		output = model(train_data)
		loss = lossfun(output, train_target)
		loss.backward()
		optimizer.step()
		if batch_idx == len(dataloader) - 1:
			#loss_list.append(loss.item())
			print(f'Train Epoch: {i} Loss: {loss.item()}')
			if args.dry_run:
				break

# valid
def valid(args, model, device, dataloader, epoch):
	with torch.no_grad():
		model.eval()
		loss_fun = nn.BCELoss()
		for batch_idx, (data, target) in enumerate(dataloader):
			data, target = data.to(device), target.to(device)
			output = model(data)

			loss = loss_fun(output, target)
			if batch_idx == len(dataloader) - 1:
				val_loss_list.append(loss.item())
				  
				# pred
				pred_all = ((output.to(cpu).detach().numpy() > 0.5) * 255) / 255

				#target
				target_all = target.to(cpu).detach().numpy()

				# tensor to numpy
				target_cpu, pred_cpu = target_all.reshape(-1), pred_all.reshape(-1)

				print(f"cm : {confusion_matrix(target_cpu, pred_cpu)}")
				acc = accuracy_score(target_cpu, pred_cpu)
				print(f"acc : {acc}")

				f1_all = f1_score(target_cpu, pred_cpu)
				print(f"f_score : {f1_all}")

				if args.dry_run:
					break

model = MLP.MLP().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=0.001)
model.to(device)

# 繰り返し数は要調整
# Lossが0.05くらいまで下がるはず（下がらなかったらやり直す）

for i in range(args.epochs):
	train(args, model=model, device=device, dataloader=train_loader, optimizer=optimizer, epoch=i)
    #valid(args, model=model, device=device, dataloader=val_loader, epoch=i)

# save to file
cpu = torch.device("cpu")
model.to(cpu)
torch.save(model.state_dict(), "train_result/sf_model.pt")

# load model from file
model_path = "train_result/sf_model.pt"

model = MLP.MLP()
model.load_state_dict(torch.load(model_path))
model.eval()

def eval_image(index, fname, thre):
	imw = IMW
	imh = IMH
	testx = np.zeros((0, 3, imh, imw), dtype=np.float32)

	a, img = load_image(fname, imw, imh)
	a1 = np.expand_dims(a,axis=0)

	testx = np.append(testx, a1, axis=0)

	t0 = time.time()
	testy = model(torch.FloatTensor(testx))
	print(f"testy : {testy.shape}")
	print(f"testy : {testy.argmax(1)}")
	print('forward time [s]: ' + str(time.time()-t0))

	imd = Image.new('RGB', (imw*3, imh))
    
	thimg_g = (testy.to(cpu).detach().numpy()[0][0] > thre) * 255
	thimg_w = (testy.to(cpu).detach().numpy()[0][1] > thre) * 255

	"""
	testy = testy.to(cpu).argmax(1).detach().numpy
	for y in range(imh):
		for x in range(imw):
			if testy[y][x] == 0:
				thimg_g[y][x] == 255
			if testy[y][x] == 1:
				thimg_w[y][x] == 255
	"""

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
	plt.show()
	plt.savefig('train_result/eval_image/eval_img_' + str(index).zfill(3) + '.png')

test_files = glob.glob('test_dataset/*.jpg')
os.makedirs('/train/train_result/eval_image', exist_ok=True)
for idx, tf in enumerate(test_files):
	eval_image(idx, tf, 0.5)
