import torch
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np


def test():
    # # Load the checkpoint
    # checkpoint = torch.load('UNetMobV2.pt', map_location=torch.device('cpu'))
    # # Print the checkpoint keys
    # print(checkpoint.keys())
    # # If the checkpoint is a dictionary and contains 'model_state_dict', use it
    # if 'model_state_dict' in checkpoint:
    #     state_dict = checkpoint['model_state_dict']
    # else:
    #     state_dict = checkpoint
    #
    # # Define the model architecture (ensure it matches the saved model's architecture)
    # model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
    # # Load the state dictionary into the model
    # model.load_state_dict(state_dict)

    model = torch.load('smoke_best_model.pt', map_location=torch.device('cpu'))
    model.eval()

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust to your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust if needed
    ])

    # Load and preprocess the image
    image_path = '16.png'
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

    # Run inference
    with torch.no_grad():
        output = model(input_batch)

    # Post-process the output
    output_np = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    segmentation_mask = np.argmax(output_np, axis=0)  # Assuming output is a probability map

    # If you need to visualize or save the result
    import matplotlib.pyplot as plt

    plt.imshow(segmentation_mask, cmap='gray')
    # image = cv2.imread(image_path)
    # masked_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(segmentation_mask))
    # plt.imshow(masked_image, cmap='rgb')
    plt.show()
