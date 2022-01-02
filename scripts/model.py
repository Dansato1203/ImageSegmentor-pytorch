import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self, n_units1, n_units2):
		super(MLP, self).__init__()
		self.l1 = nn.Conv2d(3, n_units1, 5, stride=1, padding=2)
		#self.l2 = nn.Conv2d(n_units1, n_units2, 3, stride=1, padding=1)
		self.l3 = nn.Conv2d(n_units1, 3, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = F.relu(self.l1(x))
		#x = F.relu(self.l2(x))
		x = self.l3(x)
		return torch.sigmoid(x)
