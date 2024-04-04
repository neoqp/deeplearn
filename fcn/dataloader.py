from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch

class Compose(object):
	def __init__(self, transform_data, transform_target):
		self.transform_data = transform_data
		self.transform_target = transform_target

	def __call__(self, image, target):
		for t in self.transform_data:
			image = t(image)

		for t in self.transform_target:
			target = t(target)
			
		return image, target
	
class Remove255(object):
	def __call__(self, tensor):
		return torch.where(tensor>20, torch.tensor(0), tensor)

transform_target = []
transform_target.append(transforms.Resize((224,224)))
transform_target.append(transforms.PILToTensor())
transform_target.append(Remove255())
transform_data = []
transform_data.append(transforms.Resize((224,224)))
transform_data.append(transforms.ToTensor())
transform_data.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
transform = Compose(transform_data, transform_target)

VOC2012_train = torchvision.datasets.VOCSegmentation('VOC2012_seg', image_set="train", transforms=transform)
VOC2012_test = torchvision.datasets.VOCSegmentation('VOC2012_seg', image_set="val", transforms=transform)

batch_size = 64
trainloader = DataLoader(VOC2012_train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(VOC2012_train, batch_size=batch_size, shuffle=True)