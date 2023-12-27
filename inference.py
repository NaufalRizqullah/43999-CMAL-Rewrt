import torch
import pathlib

from torchvision import transforms

from PIL import Image
from typing import Union

# image of testing
PATH_IMAGE_INFER = pathlib.Path("test/2012_Aston_Martin_V8_Vantage.jpg")

# transformation (same as training )
transform_image = transforms.Compose([
    transforms.Resize((421, 421)),
    transforms.RandomCrop(368, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5,), (0.5, 0.5, 0.5) )
])

# Model setup
TRAINED_MODEL = pathlib.Path("weights/SW_training_10.pth")
net = torch.load(TRAINED_MODEL)

# Setup class names
CLASS_NAMES_PATH = pathlib.Path("test/class_names_stanford_cars.txt")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [c.strip() for c in  f.readlines()]


def inference(
        path_image: Union[str, pathlib.Path],
        tform: transforms,
        model: torch.nn.Module,
        device: torch.device
    ):
    
    # get image to PIL format
    img = Image.open(path_image).convert("RGB")

    # turn image to same transformation and add batch size
    transformed_image  = tform(img)
    img = torch.unsqueeze(transformed_image, dim=0)
    img = img.to(device)

    # send model to the target device
    model.to(device)

    # turn to eval mode
    model.eval()

    # using inference_mode 
    with torch.inference_mode():
        # do the forward pass
        _, _, _, output_concat, _, _, _ = model(img)

        logist = output_concat
    
    # convert `logist` into `prediction probabilities`
    y_probs = torch.softmax(logist, dim=1)

    # convert `probabilities` into `prediction label`
    y_pred = torch.argmax(y_probs, dim=1)

    # do some converting label into readable label
    label = class_names[y_pred]

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(y_probs[0][i]) for i in range(len(class_names))}

    return pred_labels_and_probs, label

if __name__ == "__main__":
    # only works if using CUDA. cause TResNet using CUDA
    device = torch.device("cuda")

    probs_label, label = inference(
        path_image=PATH_IMAGE_INFER,
        tform=transform_image,
        model=net,
        device=device
    )

    print(f"Predict is: {label}")
    print("\n")
    print(probs_label)