import torch
import torchvision
import numpy as np

from torchvision import transforms, models
from torch.utils.data import DataLoader
from timeit import default_timer as timer


def cosine_anneal_schedule(t, nb_epoch, lr):
    """Cosine annealing learning rate schedule.

    Args:
        t (int): Current iteration or epoch.
        nb_epoch (int): Total number of epochs.
        lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate based on cosine annealing schedule.
    """
    cos_inner = np.pi * (t % nb_epoch)  # t - 1 is used when t has 1-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def map_generate(attention_map, pred, p1, p2):
    """Generate attention map based on predictions and learnable parameters.

    Args:
        attention_map (torch.Tensor): Input attention map tensor with dimensions (batches, feaC, feaH, feaW).
        pred (torch.Tensor): Predictions tensor with dimensions (batches, ...).
        p1 (torch.Tensor): Learnable parameter tensor with dimensions (..., feaC).
        p2 (torch.Tensor): Learnable parameter tensor with dimensions (..., feaC).

    Returns:
        torch.Tensor: Output attention map tensor with dimensions (batches, feaH, feaW).
    """
    batches, feaC, feaH, feaW = attention_map.size()

    out_map = torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        # Extract attention map for the current batch
        map_tpm = attention_map[batch_index]

        # Reshape and permute attention map and p1 tensor
        map_tpm = map_tpm.view(feaC, feaH * feaW).t()
        p1_tmp = p1.t()

        # Perform matrix multiplication and reshape the result
        map_tpm = torch.mm(map_tpm, p1_tmp).t()
        map_tpm = map_tpm.view(map_tpm.size(0), feaH, feaW)

        # Extract prediction for the current batch
        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()

        # Select the corresponding column from p2
        p2_tmp = p2[pred_ind].unsqueeze(1)

        # Reshape and permute attention map and perform another matrix multiplication
        map_tpm = map_tpm.view(map_tpm.size(0), feaH * feaW).t()
        map_tpm = torch.mm(map_tpm, p2_tmp).t()

        # Reshape the final result and assign it to the output attention map
        out_map[batch_index] = map_tpm.view(feaH, feaW)

    return out_map


def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
    """Apply attention map to images.

    Args:
        images (torch.Tensor): Input image tensor with dimensions (batches, channels, imgH, imgW).
        attention_map (torch.Tensor): Attention map tensor with dimensions (batches, imgH, imgW).
        theta (float): Threshold value for attention map.
        padding_ratio (float): Padding ratio for extracting regions of interest.

    Returns:
        torch.Tensor: Modified image tensor with attention applied.
    """
    images = images.clone()  # Clone to avoid modifying the original images tensor
    attention_map = attention_map.clone().detach()  # Clone and detach to avoid backpropagation
    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]

        # Upsample the attention map to match image dimensions
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()

        # Normalize attention map values to [0, 1]
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)

        # Threshold the attention map based on the given threshold (theta)
        map_tpm = map_tpm >= theta

        # Find non-zero indices in the thresholded attention map
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)

        # Extract region of interest in both height and width with padding
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        # Extract and upsample the region of interest in the image
        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        # Update the modified image in the images tensor
        images[batch_index] = image_tmp

    return images


def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
    """Apply combined attention maps to images.

    Args:
        images (torch.Tensor): Input image tensor with dimensions (batches, channels, imgH, imgW).
        attention_map (torch.Tensor): First attention map tensor with dimensions (batches, imgH, imgW).
        attention_map2 (torch.Tensor): Second attention map tensor with dimensions (batches, imgH, imgW).
        attention_map3 (torch.Tensor): Third attention map tensor with dimensions (batches, imgH, imgW).
        theta (float): Threshold value for combined attention map.
        padding_ratio (float): Padding ratio for extracting regions of interest.

    Returns:
        torch.Tensor: Modified image tensor with combined attention applied.
    """
    images = images.clone()  # Clone to avoid modifying the original images tensor
    attention_map = attention_map.clone().detach()  # Clone and detach to avoid backpropagation
    attention_map2 = attention_map2.clone().detach()  # Clone and detach to avoid backpropagation
    attention_map3 = attention_map3.clone().detach()  # Clone and detach to avoid backpropagation

    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]

        # Upsample and normalize the first attention map
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)

        # Upsample and normalize the second attention map
        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        # Upsample and normalize the third attention map
        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        # Combine the attention maps
        map_tpm = (map_tpm + map_tpm2 + map_tpm3)

        # Normalize the combined attention map
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)

        # Threshold the combined attention map
        map_tpm = map_tpm >= theta

        # Find non-zero indices in the thresholded combined attention map
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)

        # Extract region of interest in both height and width with padding
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        # Extract and upsample the region of interest in the image
        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        # Update the modified image in the images tensor
        images[batch_index] = image_tmp

    return images


def test_tresnetl(net, criterion, batch_size, test_path):
    """
    Evaluate the TResNet model on a test dataset.

    Args:
        net (torch.nn.Module): The TResNet model to be evaluated.
        criterion (torch.nn.Module): The loss criterion used for evaluation.
        batch_size (int): The batch size for the test data loader.
        test_path (str): The path to the test dataset.

    Returns:
        Tuple[float, float]: A tuple containing the test accuracy and test loss.
    """
    net.eval() # Set the model to evaluation mode

    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    # Define the transformation for the test dataset
    transform_test = transforms.Compose([
        transforms.Resize((421, 421)),
        transforms.CenterCrop(368),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the test dataset and data loader
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)

        # Turn off gradient computation for inference
        with torch.inference_mode():

            # Forward pass
            output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

            # Calculate attention maps
            p1 = net.state_dict()['classifier3.1.weight']
            p2 = net.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output_3, p1, p2)

            p1 = net.state_dict()['classifier2.1.weight']
            p2 = net.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output_2, p1, p2)

            p1 = net.state_dict()['classifier1.1.weight']
            p2 = net.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output_1, p1, p2)

            # Generate attention-guided inputs
            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

            # Combine model outputs
            outputs_com2 = output_1 + output_2 + output_3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

            # Calculate loss
            loss = criterion(output_concat, targets)

            # Update metrics
            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            _, predicted_com2 = torch.max(outputs_com2.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()
            correct_com2 += predicted_com2.eq(targets.data).cpu().sum()


    # Calculate and return test accuracy and test loss
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss


def timing_decorator(func):
    """
    A decorator to measure the execution time of a function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.

    Example:
        @timing_decorator
        def example_function():
            # Your code here
            pass

        # Call the decorated function
        example_function()
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper function that calculates the execution time of the decorated function.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            The result of the decorated function.
        """
        start_time = timer()
        result = func(*args, **kwargs)
        end_time = timer()
        print(f"[INFO] {func.__name__} execution time: {end_time - start_time:.3f} seconds")
        return result
    return wrapper