from model import Model
from dataset import DCMImageDataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output
pd.set_option('mode.chained_assignment',  None)
data_dir = "../../dataset/rsna-2024-lumbar-spine-degenerative-classification/"

coordinates = pd.read_csv(data_dir + 'train_label_coordinates.csv')
descriptions = pd.read_csv(data_dir + 'train_series_descriptions.csv')
train = pd.read_csv(data_dir + 'train.csv')

d = DCMImageDataset(series='Axial T2',
                          coordinates_file=coordinates,
                          descriptions_file=descriptions,
                          train_file=train,
                          img_dir=data_dir,
                          )

model = Model()
model = model.to('cuda')

lr = 10**-3
optim = torch.optim.Adam(params=model.parameters(), lr=lr)
epoch = 3
loss_fn = nn.CrossEntropyLoss()

dataloader = DataLoader(d, batch_size=256, shuffle=True, drop_last=True)

save_path = "weights/1.pt"
model.load_state_dict(torch.load(save_path))

losses = []
for epoch_cnt in range(epoch):
    loss_sum = 0

    for i, (input, answer) in enumerate(dataloader):
        optim.zero_grad()

        input = input.to('cuda').unsqueeze(1)
        answer = answer[:, 0].type(torch.LongTensor).to('cuda')
        output = model(input).to('cuda')

        loss = loss_fn(output, answer)
        loss.backward()

        optim.step()
        
        loss_sum += loss.item()
        if i%100==99:
            print(f"iter : {i+1}, loss = {round(loss_sum / 100, 4)}")
            losses.append(loss_sum / 10)
            loss_sum = 0
            torch.save(model.state_dict(), save_path)
            '''
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.xlabel('Batch number')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.show()
            '''