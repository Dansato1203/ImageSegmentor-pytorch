import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import cv2
from PIL import Image
import numpy as np
import argparse
import glob
import os

IMW = 320
IMH = 240

import matplotlib.pyplot as plt

import MLP

def load_image(fname, imw, imh):
	img = Image.open(fname).resize((imw, imh))

	a = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
	return a, img

def png2array(lfile_array, imw, imh):
	a = np.zeros((3, imh, imw), dtype=np.float32)
	for x in range(imw):
		for y in range(imh):
			if (lfile_array[y][x] == 50):
				a[0][y][x] = 1
			elif (lfile_array[y][x] == 225):
				a[1][y][x] = 1
	return a

class SoccerFieldDataset(Dataset):
	def __init__(self, image_dir, transform=None):
		imw = IMW
		imh = IMH
		x = np.zeros((0, 3, imh, imw), dtype=np.float32)
		t = np.zeros((0, 3, imh, imw), dtype=np.float32)
		jpgfiles = glob.glob(image_dir + '*.jpg')
		print(jpgfiles)
		for f in jpgfiles:
			#plt.figure()
			a, img = load_image(f, imw, imh)
			a1 = np.expand_dims(a,axis=0)
			x = np.append(x, a1, axis=0)
           
			#plt.imshow(img)
            
			lfile = os.path.splitext(f)[0] + '_label.png'
			print(lfile)
			img = Image.open(lfile).resize((imw, imh))
            
			#plt.figure()
			#plt.imshow(img)
            
			a = np.asarray(img).astype(np.float32)
			a = a.astype(np.int32)
			#print(np.sum(a))
			a = png2array(a, imw, imh)
			a1 = np.expand_dims(a,axis=0)
			t = np.append(t, a1, axis=0)
        
		self.data  = x
		self.label = t
		#print(self.data.shape)
		#print(self.label.shape)

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		#print(self.data.shape)
		#print(self.label.shape)
		images = self.data[idx, :, :, :]
		labels = self.label[idx, :, :]

		#print(f"images: {images}")
		#rint(f"labels: {labels}")

		return (images, labels)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
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


print('Dataset_dir : ', end='')
Dataset_dir = input()
print('val_dataset : ', end='')
val_dataset_dir = input()

dataset = SoccerFieldDataset(Dataset_dir + '/')
val_dataset = SoccerFieldDataset(val_dataset_dir + '/')
train_loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=0)

cpu = torch.device("cpu")

# exec training
def train(args, model, device, dataloader, optimizer, epoch):
	model.train()
	lossfun = nn.BCELoss()
	for batch_idx, (train_data, train_target) in enumerate(dataloader):
		train_data, train_target = train_data.to(device), train_target.to(device)
		optimizer.zero_grad()
		output = model(train_data)
		loss = lossfun(output, train_target)
		loss.backward()
		optimizer.step()
		if batch_idx == len(dataloader) - 1:
			loss_list.append(loss.item())
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
				pred_g = ((output.to(cpu).detach().numpy()[:][0] > 0.5) * 255) / 255
				pred_w = ((output.to(cpu).detach().numpy()[:][1] > 0.5) * 255) / 255

				#target
				target_all = target.to(cpu).detach().numpy()
				target_g = target.to(cpu).detach().numpy()[:][0]
				target_w = target.to(cpu).detach().numpy()[:][1]

				# tensor to numpy
				target_cpu, pred_cpu = target_all.reshape(-1), pred_all.reshape(-1)
				target_g, pred_g = target_g.reshape(-1), pred_g.reshape(-1)
				target_w, pred_w = target_w.reshape(-1), pred_w.reshape(-1)

				print(f"cm : {confusion_matrix(target_cpu, pred_cpu)}")
				acc = accuracy_score(target_cpu, pred_cpu)
				acc_g = accuracy_score(target_g, pred_g)
				acc_w = accuracy_score(target_w, pred_w)
				print(f"acc : {acc}")
				print(f"acc_g : {acc_g}")
				print(f"acc_w : {acc_w}")
				acc_list.append(acc.item())
				acc_g_list.append(acc_g.item())
				acc_w_list.append(acc_w.item())

				f1_all = f1_score(target_cpu, pred_cpu)
				f1_g = f1_score(target_g, pred_g)
				f1_w = f1_score(target_w, pred_w)
				print(f"f_score : {f1_all}")
				f_list.append(f1_all.item())
				f_g_list.append(f1_g.item())
				f_w_list.append(f1_w.item())

				if args.dry_run:
					break

model = MLP.MLP(4, 3).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=0.001)
model.to(device)

# 繰り返し数は要調整
# Lossが0.05くらいまで下がるはず（下がらなかったらやり直す）
loss_list = []
val_loss_list = []
acc_list = []
acc_g_list = []
acc_w_list = []
f_list = []
f_g_list = []
f_w_list = []

for i in range(1000):
	train(args, model=model, device=device, dataloader=train_loader, optimizer=optimizer, epoch=i)
	valid(args, model=model, device=device, dataloader=val_loader, epoch=i)

	plt.figure(figsize=(16,15))
	plt.subplot(3,1,1)
	plt.plot(range(i+1), loss_list, 'r', label='train_loss', linewidth=2)
	plt.plot(range(i+1), val_loss_list, 'b', label='val_loss', linewidth=2)
	plt.legend(fontsize=15)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid(True)
	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 1000, 0, ymax])
	plt.xticks(range(0, 1000 + 1, 100))

	plt.subplot(3,1,2)
	plt.plot(range(i+1), acc_list, 'r', label='all_accuracy', linewidth=2)
	plt.plot(range(i+1), acc_g_list, 'g', label='green_accuracy', linewidth=2)
	plt.plot(range(i+1), acc_w_list, 'k', label='white_accuracy', linewidth=2)
	plt.legend(fontsize=15)
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid(True)
	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 1000, 0, ymax])
	plt.xticks(range(0, 1000 + 1, 100))

	plt.subplot(3,1,3)
	plt.plot(range(i+1), f_list, 'r', label='all_f-score', linewidth=2)
	plt.plot(range(i+1), f_g_list, 'b', label='green_f-score', linewidth=2)
	plt.plot(range(i+1), f_w_list, 'y', label='white_f-score', linewidth=2)
	plt.legend(fontsize=15)
	plt.xlabel('epoch')
	plt.ylabel('f_score')
	plt.grid(True)
	[xmin, xmax, ymin, ymax] = plt.axis()
	plt.axis([0, 1000, 0, ymax])
	plt.xticks(range(0, 1000 + 1, 100))

	plt.pause(.01)

# save to file
cpu = torch.device("cpu")
model.to(cpu)
torch.save(model.state_dict(), "wl_model.pt")
