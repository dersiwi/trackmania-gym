

import os, sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neuronal_networks.vision_encoder.conv_NNs import VisionModelEight
from neuronal_networks.pretraining.dataloader import LateralDistanceDataset
from tqdm import tqdm


class PredictorModel(nn.Module):
    """This only adds two fully connected layrs after the vision model to create a predictor."""
    def __init__(self):
        super().__init__()

        self.flattened_size = 64 
        self.vision_model = VisionModelEight(in_color_channels=1, out_dim=self.flattened_size) # <---------------------------------------------PUT VISION MODEL TO PRETRAIN HERE 
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.vision_model(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)


def train(
    data_dir,
    batch_size=32,
    epochs=10,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="cnn_pretrained.pth"
):
    dataset = LateralDistanceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = PredictorModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            preds = model(images).squeeze()
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), 
                   )

    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    train(r"C:\Users\siwis\Documents\dataset")