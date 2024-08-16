from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model.unet import UNet
from model.diffusion import DiffusionModel
from utils import make_grid

def transform(examples):
    jitter = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
     ])
    image_tensors = [jitter(image.convert("RGB")) for image in examples["image"]]
    return {'img_input': image_tensors}

def main():
    # load model
    model = UNet(in_dim=64,
                 dim_mults = (1, 2, 4, 8, 16),
                 is_attn = (False, False, False, True, True)
                 )
    diffusion = DiffusionModel(model = model,
                               num_timesteps=1_000)
    print(diffusion)
    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='../../dataset', train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-5)
    diffusion.to(torch.device("cuda"))

    best_loss = 100
    for epoch in range(1_000):
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
        print(f"train_loss: {train_avg_loss}, lr: {2e-5}")
        loss_total = 0
        # eval
        if train_avg_loss < best_loss:
            best_loss = train_avg_loss
            with torch.no_grad():
                x = diffusion.sample(16,3,128)
            imgs_grid = make_grid(x, 4, 4)
            imgs_grid.save(f"img/{epoch}.png")
            torch.save(diffusion.state_dict(), f"lastmodel.pt")


if __name__ == "__main__":
    main()