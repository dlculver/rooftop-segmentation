"""
script for various utilities
"""
import os
from pathlib import Path
import numpy as np
import torch

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt

from data import RoofDataSet, open_image


def pred_to_pil(pred):
    """
    Takes a prediction from UNet model and returns PIL IMAGE for displaying later

    TODO: need to adapt to also visualize whole batches. Currently only good for batch_size = 1
    """
    # Assuming 'label' is the tensor you obtained from the Unet model's prediction

    # Step 1: Detach the tensor and move it to the CPU
    pred_detached = pred.detach().cpu()

    # Step 2: Squeeze the tensor to remove the batch dimension
    pred_squeezed = pred_detached.squeeze(0)

    # Step 3: Convert the squeezed tensor to a NumPy array
    pred_np = pred_squeezed.numpy()

    # Step 4: Rescale the values to the range [0, 255]
    pred_np = (pred_np * 255).astype(np.uint8)

    # Step 5: Convert the NumPy array back to a PIL Image
    pred_pil = Image.fromarray(pred_np[0], mode="L")

    return pred_pil


def display_trio(original, label, prediction):
    """
    display the original photo, ground truth label, and prediction of model in one line.
    for train sets
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # plot original
    axes[0].imshow(original)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    # plot label
    axes[1].imshow(label)
    axes[1].axis("off")
    axes[1].set_title("Ground Truth Label")

    # plot prediction
    axes[2].imshow(prediction)
    axes[2].axis("off")
    axes[2].set_title("Model prediction")

    plt.show()

    # Close the figure to release memory
    plt.close(fig)


def display_duo(image, prediction):
    """
    display an image and the prediction of the model.
    for test sets
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # plot image
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Image")

    # plot prediction
    axes[1].imshow(prediction)
    axes[1].axis("off")
    axes[1].set_title("Model Prediction")

    plt.show()

    plt.close(fig)


def show_predictions(
    dataset: RoofDataSet, model, binary_pred=False, batch_size=1, max_num=3
):
    """
    utility to display the original image, label, and prediction

    Arguments:
        dataset (RoofDataSet) : dataset made from Dida image directories
        model (nn.Module) : model making the predictions
        batch_size : int, defaults to 1
        max_num (int) : number of images and predictions to show
    """

    mode = dataset.mode

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set the model to evaluation mode
    model.eval()

    nb_shown = 0

    # Iterate through the dataloader and visualize each image and its prediction

    if mode == "train" or mode == "val":
        for batch in dataloader:
            (
                images,
                labels,
            ) = batch  # have shape (batch_size, 3, 256, 256) and (batch_size, 1, 256, 256)

            if nb_shown == max_num:
                break
            # Assuming your model returns the prediction as a tensor
            with torch.no_grad():
                predictions = model(images)
                if binary_pred:
                    predictions = (predictions > 0.5).float()

            # Visualize the images and predictions in one row
            for i in range(images.shape[0]):
                if dataset.normalize:
                    images[i] = dataset.inverse_normalize(images[i])
                original_image = ToPILImage()(images[i])
                label = ToPILImage()(labels[i].squeeze())  # squeeze since gray scale
                prediction = pred_to_pil(predictions)

                display_trio(original_image, label, prediction)

                nb_shown += 1

    if mode == "test":
        for batch in dataloader:
            images = batch  # images shape: (bs, 3, 256, 256)

            if nb_shown == max_num:
                break

            with torch.no_grad():
                predictions = model(images)
                if binary_pred:
                    predictions = (predictions > 0.5).float()

            for i in range(images.shape[0]):
                if dataset.normalize:
                    images[i] = dataset.inverse_normalize(images[i])
                image = ToPILImage()(images[i])
                predictions = pred_to_pil(predictions)

                display_duo(image, predictions)

                nb_shown += 1

def save_predictions(input_dir, output_dir, model):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    img_mean = [0.485, 0.456, 0.406]  # from pretrained MobileNet
    img_std = [0.229, 0.224, 0.225]  # from pretrained MobileNet

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    paths = sorted(
            [p for p in input_dir.iterdir() if p.is_file()]
        ) 
    images = [(open_image(p), p.name) for p in paths]

    model.eval()
    with torch.no_grad():
        for image_pil, name in images:
            
            image_pil = image_pil.convert('RGB')
            image_tensor = TF.to_tensor(image_pil)
            image_tensor = torch.unsqueeze(image_tensor, 0)

            image_tensor = TF.normalize(image_tensor, img_mean, img_std)
            prediction = model(image_tensor)
            prediction = (prediction > 0.5).float()

            prediction = pred_to_pil(prediction)
            

            output_path = os.path.join(output_dir, name)

            prediction.save(output_path)

    print(f'Saved predictions!')