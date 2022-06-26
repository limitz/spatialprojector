import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.io import write_png
from torchvision.utils import make_grid

from spatial import SpatialTransformer, SpatialProjector

# based on the Spatial Transformer Network Tutorial here: 
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

class Example:
	def __init__(self):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.norm = (0.1307, 0.3081)
		t = T.Compose((
			T.ToTensor(),
			T.Normalize((self.norm[0],), (self.norm[1],)),
			T.RandomPerspective(0.3, p=1)
			))

		dataset_training = MNIST("./data", train=True, transform=t, download=True)
		dataset_validate = MNIST("./data", train=False, transform=t, download=True)
		self.loader_training = DataLoader(dataset_training, batch_size=64, shuffle=True)
		self.loader_validate = DataLoader(dataset_validate, batch_size=64, shuffle=True)

		self.spatial = SpatialProjector(1, (28,28))
		#self.spatial = SpatialTransformer(1, (28,28))
		
		self.model = nn.Sequential(
			self.spatial,

			nn.Conv2d(1,10,5),  # 10 x 24 x 24
			nn.MaxPool2d(2),    # 10 x 12 x 12
			nn.LeakyReLU(),
			
			nn.Conv2d(10,20,5), # 20 x 8 x 8
			nn.MaxPool2d(2),    # 20 x 4 x 4
			nn.LeakyReLU(),
			nn.Flatten(),       # 320
			nn.Linear(320,50),  # 50
			nn.LeakyReLU(),
			nn.Linear(50, 10),  # 10
			nn.LogSoftmax(1)).to(self.device)

		self.lossFunction = nn.NLLLoss().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

	def epoch(self, idx):
		for batchIdx, (img, gt) in enumerate(self.loader_training):
			img, gt = img.to(self.device), gt.to(self.device)
			
			self.optimizer.zero_grad()
			pred = self.model(img)
			loss = self.lossFunction(pred, gt)
			loss.backward()
			self.optimizer.step()

			if batchIdx % 100 == 0:
				print(f"EPOCH {idx}:  {batchIdx} / {len(self.loader_training)} {loss.item()}")

	def validate(self):
		total_loss = 0
		score = 0
		with torch.no_grad():
			self.model.eval()
			
			img,_ = next(iter(self.loader_validate))
			img = img.to(self.device)
			pred = self.spatial(img) * self.norm[1] + self.norm[0]
			img = img * self.norm[1] + self.norm[0]

			write_png(make_grid((img*255).byte().cpu(), 8), "original.png")
			write_png(make_grid((pred*255).byte().cpu(), 8), "transformed.png")
			
			for img, gt in self.loader_validate:
				img = img.to(self.device)
				gt = gt.to(self.device)
				pred = self.model(img)
				total_loss += self.lossFunction(pred, gt).item()
				m = pred.max(1, keepdim=True)[1]
				score += m.eq(gt.view_as(m)).sum().item()
		
		total_loss /= len(self.loader_validate)
		score /= len(self.loader_validate) * self.loader_validate.batch_size
		score *= 100
		print(f"Average loss: {total_loss}, Score: {score}%")

e = Example()
for i in range(10):
	e.epoch(i)
	e.validate()
