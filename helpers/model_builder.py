import torch
import os
import requests
from typing import Dict, List
from pathlib import Path

from src.models.tresnet_v2 import TResnetL_V2
from layers import network_wrapper


def create_net_layers(model_params: Dict, download_pretrained_tresnet: bool = True) -> List:
    """Create layers for TResNet-L V2 model.

    Args:
        model_params (Dict): Dictionary containing model parameters.
        download_pretrained_tresnet (bool, optional): Whether to download pretrained TResNet-L V2 weights.
            Defaults to True.

    Returns:
        List: List of model layers for NetworkWrapper.
    """
    # Checking if the model path of `src` TResNet already exists
    dir_tresnet = Path("./src")
    assert dir_tresnet.is_dir(), f"The directory '{dir_tresnet}' does not exist."

    # Instantiate the TResNet-L V2 model
    model = TResnetL_V2(model_params)

    # Download model weights if not already present
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    url_weights = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/stanford_cars_tresnet-l-v2_96_27.pth'
    weight_path = weights_dir / "tresnet-1-v2.pth"

    if download_pretrained_tresnet:
        if not weight_path.exists():
            print("[INFO] Downloading Weights TResNet-L V2...")
            r = requests.get(url_weights)

            with open(weight_path, "wb") as code:
                code.write(r.content)

        # Set pretrained weights for the model
        pretrained_weights = torch.load(weight_path)
        model.load_state_dict(pretrained_weights["model"])

    # Set the layers for NetworkWrapper
    net_layers = list(model.children())[0].children()
    net_layers = list(net_layers)

    return net_layers


def create_model(params: Dict, download_weight: bool = True):
    """Create a model using specified parameters.

    Args:
        params (Dict): Dictionary containing model parameters.
        download_weight (bool, optional): Whether to download pretrained TResNet-L V2 weights.
            Defaults to True.

    Returns:
        NetworkWrapper: Instantiated model.
    """
    layers = create_net_layers(model_params=params, download_pretrained_tresnet=download_weight)
    model = network_wrapper.NetworkWrapper(net_layers=layers, num_classes=params["num_classes"])
    print(f"[INFO] COmplete Creating Model CMAL.")
    return model

