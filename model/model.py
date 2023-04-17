import torch
import yaml
from torch.utils.data import DataLoader
from yolov5.datasets import YoloDataset
from yolov5.models import YoloV5
from yolov5.utils import train_utils

# Load the YAML configuration file
with open("yolov5/data/custom.yaml", "r") as f:
    cfg = yaml.load(f)

# Create the dataset
train_dataset = YoloDataset(
    img_dir=cfg["train"]["img_dir"],
    label_dir=cfg["train"]["label_dir"],
    img_size=cfg["img_size"],
    augment=True,
    multiscale=cfg["multiscale"],
    normalized_labels=cfg["normalized_labels"],
)

# Create the data loader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=train_utils.collate_fn,
)

# Create the model
model = YoloV5(cfg["model"]["backbone"], cfg["num_classes"])

# Define the loss function
criterion = torch.nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

# Train the model
for epoch in range(cfg["epochs"]):
    for batch_idx, (imgs, targets) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(imgs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        if batch_idx % cfg["print_interval"] == 0:
            print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch+1, cfg["epochs"], batch_idx+1, len(train_dataloader), loss.item()))

# Save the trained model
torch.save(model.state_dict(), "vehicle_damage_detection.pth")
