from pathlib import Path
from helpers.engine import *
from helpers.model_builder import *

# Prepare Dataset
DATASET_PATH = Path("dataset")

transform_train = transforms.Compose([
    transforms.Resize((421, 421)),
    transforms.RandomCrop(368, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5,), (0.5, 0.5, 0.5) )
])

train_dataset = datasets.ImageFolder(
    root= DATASET_PATH / "train",
    transform=transform_train
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Prepare Model
layer_params = {'num_classes': 196}
model = create_model(params=layer_params, download_weight=True)

if __name__ == "__main__":
    # train()
    pass