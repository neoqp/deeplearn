from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model.unet import UNet
from model.diffusion import DiffusionModel
from utils import make_grid

def main():
    # load model
    model = UNet(in_dim=64,
                 dim_mults = (1, 2, 4, 8, 16),
                 is_attn = (False, False, False, True, True)
                 )
    diffusion = DiffusionModel(model = model,
                               num_timesteps=1_000)
    # print(diffusion)
    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='../../dataset', train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
    diffusion.to(torch.device("cuda"))

    best_loss = 100
    #diffusion.load_state_dict(torch.load('lastmodel.pt'))
    for epoch in range(0, 1_000):
        # train
        print(f"{epoch}th epoch training...")
        loss_total = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            data = batch[0].to("cuda")
            optimizer.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optimizer.step()
            loss_total += loss
        train_avg_loss = loss_total/len(train_dataloader)
        print(f"train_loss: {train_avg_loss}, lr: {scheduler.get_last_lr()}")
        loss_total = 0
        scheduler.step()
        # eval
        if train_avg_loss < best_loss and epoch%10==0:
            best_loss = train_avg_loss
            with torch.no_grad():
                x = diffusion.sample(16,3,32)
            imgs_grid = make_grid(x, 4, 4)
            imgs_grid.save(f"img/{epoch}.png")
            torch.save(diffusion.state_dict(), f"lastmodel.pt")


if __name__ == "__main__":
    main()