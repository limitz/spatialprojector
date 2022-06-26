import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.io import write_png, read_image
from torchvision.utils import make_grid

from spatial import SpatialTransformer, SpatialProjector, perspective_project

# Simulates a dataset where the corners of rectangles under a 
# perspective transform are annotated with a 4 point polygon.
class ExampleDataset(Dataset):
	def __len__(self):
		return 640

	def __getitem__(self, idx):
		img = read_image("template.png") / 255
		img = img.mean(dim=0, keepdims=True)
		img = TF.resize(img, (178,256))
		c,h,w = img.shape
		src, dst = T.RandomPerspective.get_params(h, w, 0.5)
		imgt = TF.perspective(img, src, dst)

		proj = SpatialProjector.get_matrix(torch.tensor(src), torch.tensor(dst))
		orig = torch.tensor([[0,0], [w,0],[w,h],[0,h]]).float() 
		poly = perspective_project(orig, proj)
		return imgt, poly

class Example:
	def __init__(self):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		
		dataset = ExampleDataset() 

		self.loader= DataLoader(dataset, batch_size=1, shuffle=True)

		self.model = SpatialProjector(1, (178,256)).to(self.device)
		self.loss_function = nn.MSELoss().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

	def epoch(self, idx):
		for batch_idx, (imgt, poly) in enumerate(self.loader):
			n,c,h,w = imgt.shape
			size = torch.tensor([w,h]).to(self.device)
			imgt = imgt.to(self.device)
			poly = 2 * poly.to(self.device) / (size) - 1
			
			orig = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], 
					device=self.device, dtype=torch.float32)
			
			# TODO make get_matrix handle batches, now only does batch size 1
			gt = SpatialProjector.get_matrix(orig,poly[0]).to(self.device)

			self.optimizer.zero_grad()
			pred,theta = self.model(imgt * 2 - 1, include_theta=True)
			pred = pred * 0.5 + 0.5
			loss = self.loss_function(theta, gt)
			loss.backward()
			self.optimizer.step()

			if batch_idx % 10 == 0:
				print(f"EPOCH {idx}:  {batch_idx} / {len(self.loader)} {loss.item()}")
				write_png((make_grid(imgt) * 255).byte().cpu(), "transformed.png")
				write_png((make_grid(pred) * 255).byte().cpu(), "prediction.png")
e = Example()
for i in range(30):
	e.epoch(i)
	# no validation set needed as the dataset generates random samples continuously
