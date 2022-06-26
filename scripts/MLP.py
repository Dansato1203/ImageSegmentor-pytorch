import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.l1 = nn.Conv2d(3, 4, 5, stride=1, padding=2)
		self.l2 = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = nn.Linear(3, 3)
		return x
