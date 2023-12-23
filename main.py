from pathlib import Path
from helpers.engine import *
from helpers.model_builder import *

DATA_PATH_TRAIN = "./dataset"
DATASET_PATH = Path("dataset")
PRETRAIN_MODEL = Path("weights/SW_training_10.pth")

# Prepare Dataloader
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
    train(
        model=model,
        dataLoader=train_loader,
        nb_epoch=200,
        batch_size=16,
        store_name='Results_Stanford_Cars_TResNet_L',
        start_epoch=0,
        resume=False,
        model_path=PRETRAIN_MODEL,
        data_path=DATA_PATH_TRAIN
    )