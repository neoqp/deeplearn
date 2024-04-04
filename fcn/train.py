from model import FCN
import torchvision
import torch
import torch.nn as nn
from dataloader import trainloader

vgg16 = torchvision.models.vgg16(weights=True)

model = FCN()
model.to('cuda')
model.copy_params_from_vgg16(vgg16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 10**-1
momentum = 0.9
weight_decay = 2**-4
optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
epoch = 100

loss_fn = nn.CrossEntropyLoss()
save_path = 'fcn/final19.pt'
model.load_state_dict(torch.load(save_path))

for epoch_cnt in range(epoch):

	for index, (data, target) in enumerate(trainloader):
		optim.zero_grad()
		data = data.to('cuda')
		target = target.to('cuda')
		target = torch.squeeze(target)
		target = target.long()
		prediction = model(data)

		loss = loss_fn(prediction, target)
		loss.backward()
		optim.step()
	torch.save(model.state_dict(), save_path)
	print(f"finish epoch : {epoch_cnt} (lr = {lr})")
	print(f"loss : {loss}")