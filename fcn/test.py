from model import FCN
import torch
from dataloader import trainloader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = 'fcn/final21_lr-1234.pt'
model = FCN().to(device)
model.load_state_dict(torch.load(save_path))
with torch.no_grad():
    for index, (input, output) in enumerate(trainloader):
        input = input.to('cuda')
        output = output.to('cuda')
        prediction = model(input)
        prediction = prediction.cpu()
        prediction = prediction[0]
        prediction = torch.argmax(prediction, dim=0)

        cmap = plt.cm.get_cmap('tab20', 20)
        plt.subplot(121)
        for i in range(20):
            plt.imshow(prediction, cmap=cmap, alpha=0.5)
        plt.subplot(122)
        for i in range(20):
            plt.imshow(output.cpu()[0].squeeze(), cmap=cmap, alpha=0.5)
        break