import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

vgg16 = torchvision.models.vgg16(weights=True)

class FCN(nn.Module):
	def __init__(self, n_class=21):
		super(FCN, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
		self.relu1_1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		
		self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.relu2_2 = nn.ReLU(inplace=True)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

		self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.relu3_2 = nn.ReLU(inplace=True)
		self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.relu3_3 = nn.ReLU(inplace=True)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

		self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.relu4_3 = nn.ReLU(inplace=True)
		self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

		self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.relu5_1 = nn.ReLU(inplace=True)
		self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.relu5_2 = nn.ReLU(inplace=True)
		self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.relu5_3 = nn.ReLU(inplace=True)
		self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

		self.conv6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1, padding=0)
		self.relu6 = nn.ReLU(inplace=True)
		self.drop6 = nn.Dropout2d()

		self.conv7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)
		self.relu7 = nn.ReLU(inplace=True)
		self.drop7 = nn.Dropout2d()

		self.conv8 = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1, padding=0)


		self.x2_upsample = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4, stride=2)
		self.x8_upsample = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=16, stride=8)
		#self.x16_upsample = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=17, stride=16, padding=1, output_padding=1)
		#self.x32_upsample = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=33, stride=32, padding=1, output_padding=1)


		self.pool4_pred = nn.Conv2d(in_channels=512, out_channels=n_class, kernel_size=1, padding=0)
		self.pool3_pred = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1, padding=0)

	def forward(self, x):
		x = self.relu1_1(self.conv1_1(x))
		x = self.relu1_2(self.conv1_2(x))
		x = self.maxpool1(x)

		x = self.relu2_1(self.conv2_1(x))
		x = self.relu2_2(self.conv2_2(x))
		x = self.maxpool2(x)

		x = self.relu3_1(self.conv3_1(x))
		x = self.relu3_2(self.conv3_2(x))
		x = self.relu3_3(self.conv3_3(x))
		x = self.maxpool3(x)
		pool3_pred = self.pool3_pred(x)

		x = self.relu4_1(self.conv4_1(x))
		x = self.relu4_2(self.conv4_2(x))
		x = self.relu4_3(self.conv4_3(x))
		x = self.maxpool4(x)
		pool4_pred = self.pool4_pred(x)
		
		x = self.relu5_1(self.conv5_1(x))
		x = self.relu5_2(self.conv5_2(x))
		x = self.relu5_3(self.conv5_3(x))
		x = self.maxpool5(x)


		x = self.drop6(self.relu6(self.conv6(x)))
		x = self.drop7(self.relu7(self.conv7(x)))
		x = self.conv8(x)

		us = self.x2_upsample(x)
		us = us[ :, : , 1:-1, 1:-1]
		#FCN_32s = self.x32_upsample(x)
		
		us_with_pool4_prediction = us + pool4_pred
		#FCN_16s = self.x16_upsample(us_with_pool4_prediction)
		
		us2 = self.x2_upsample(us_with_pool4_prediction)
		us2 = us2[ :, : , 1:-1, 1:-1]
		us2_with_pool3_prediction = us2 + pool3_pred
		FCN_8s = self.x8_upsample(us2_with_pool3_prediction)
		result = FCN_8s[ :, :, 4:-4, 4:-4]

		return result
	
	def copy_params_from_vgg16(self, vgg16):
		features = [
			self.conv1_1, self.relu1_1,
			self.conv1_2, self.relu1_2,
			self.maxpool1,
			self.conv2_1, self.relu2_1,
			self.conv2_2, self.relu2_2,
			self.maxpool2,
			self.conv3_1, self.relu3_1,
			self.conv3_2, self.relu3_2,
			self.conv3_3, self.relu3_3,
			self.maxpool3,
			self.conv4_1, self.relu4_1,
			self.conv4_2, self.relu4_2,
			self.conv4_3, self.relu4_3,
			self.maxpool4,
			self.conv5_1, self.relu5_1,
			self.conv5_2, self.relu5_2,
			self.conv5_3, self.relu5_3,
			self.maxpool5,
		]
		for l1, l2 in zip(vgg16.features, features):
			print(l1)
			if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
				assert l1.weight.size() == l2.weight.size()
				assert l1.bias.size() == l2.bias.size()
				l2.weight.data.copy_(l1.weight.data)
				l2.bias.data.copy_(l1.bias.data)
		
		for layer in features:
			for param in layer.parameters():
				param.requires_grad = False
		
		