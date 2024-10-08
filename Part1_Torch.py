from PIL import Image, ImageDraw, ImageOps
import os
import numpy as np
import cv2

import torch
from torch import nn, optim
from torch.nn import functional
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Architecture of the model

class Conv3k(nn.Module):
    def __init__(self, channels_in, channels_out):
        
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):

        return self.conv1(x)


class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):

        super().__init__()
        self.double_conv = nn.Sequential(
                           Conv3k(channels_in, channels_out),
                           nn.BatchNorm2d(channels_out),
                           nn.LeakyReLU(),

                           Conv3k(channels_out, channels_out),
                           nn.BatchNorm2d(channels_out),
                           nn.LeakyReLU()
                            )

    def forward(self, x):

        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, channels_in, channels_out):

        super().__init__()
        self.encoder = nn.Sequential(
                        nn.MaxPool2d(2, 2),
                        DoubleConv(channels_in, channels_out)
                        )

    def forward(self, x):
        
        return self.encoder(x)


class UpConv(nn.Module):
    def __init__(self, channels_in, channels_out):

        super().__init__()
        self.upsample_layer = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bicubic'),
                        nn.Conv2d(channels_in, channels_in // 2, kernel_size = 1, stride = 1)
                        )
        self.decoder = DoubleConv(channels_in, channels_out)

    def forward(self, x1, x2):
        
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.decoder(x)


class Unet(nn.Module):
    def __init__(self, channels_in, channels, num_classes):

        super().__init__()
        self.first_conv = DoubleConv(channels_in, channels)
        self.down_conv1 = DownConv(channels, 2 * channels)
        self.down_conv2 = DownConv(2 * channels, 4 * channels)
        self.down_conv3 = DownConv(4 * channels, 8 * channels)

        self.middle_conv = DownConv(8 * channels, 16 * channels)

        self.up_conv1 = UpConv(16 * channels, 8 * channels)
        self.up_conv2 = UpConv(8 * channels, 4 * channels)
        self.up_conv3 = UpConv(4 * channels, 2 * channels)
        self.up_conv4 = UpConv(2 * channels, channels)

        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size = 1, stride = 1)

    def forward(self, x):

        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)

        x5 = self.middle_conv(x4)

        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)

        return self.last_conv(u4)

# SMTP configuration
smtp_server = 'smtp.gmail.com'
smtp_port = 587
#sender_email = 'medium.aigela@gmail.com'
#sender_password = 'ydml csoh kdbw izbm'  # Use the generated App Password here
sender_email = 'medium.aigela@gmail.com'
sender_password = 'ydml csoh kdbw izbm'  
receiver_email = ["i.m.popov03@gmail.com"]

body = 'Smoke, potentially from a forest fire, has been detected near your area through satellite observations from Platform 1!'
subject = 'Smoke Detected!'

email_signature = """
Best regards,
Fire Detection System
EnduroSat Team
"""

# Create email message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = ','.join(receiver_email)
msg['Subject'] = subject
body_with_signature = f"{body}\n\n{email_signature}"
msg.attach(MIMEText(body_with_signature, 'plain'))

# Load models
model_forest = Unet(3, 64, 2)  # Ensure Unet is properly defined or imported
model_forest.load_state_dict(torch.load(r"model_final.pt",weights_only=True))
model_forest.eval()

model_smoke = torch.load('smoke_best_model.pt', map_location='cpu',weights_only=False)
model_smoke.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_path, output_folder):
    # Load the image
    image = Image.open(image_path)

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Move to CPU
    input_batch = input_batch.to('cpu')
    model_forest.to('cpu')

    # Get the output from the model
    with torch.no_grad():
        output_smoke = model_smoke(input_batch)
        output = model_forest(input_batch)
        output = output[0]

        ML_output = output.argmax(0).byte().cpu().numpy()
        ML_output = (ML_output * 255).astype(np.uint8)
        ML_output = Image.fromarray(ML_output, mode='L')

        output_np_smoke = output_smoke.squeeze().cpu().numpy()
        segmentation_mask = np.argmax(output_np_smoke, axis=0)
        segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
        segmentation_mask = Image.fromarray(segmentation_mask, mode='L')

    # Find contours
    segmentation_mask_np = np.array(segmentation_mask)
    contours, _ = cv2.findContours(segmentation_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the original image to OpenCV format
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Draw contours
    cv2.drawContours(image_np, contours, -1, (0, 0, 255), 2)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    sum_pixels = np.sum(segmentation_mask_np)
    max_sum = 255 * segmentation_mask_np.size

    if sum_pixels / max_sum > 0.02:
        _, img_bytes = cv2.imencode('.jpg', image_np)
        img = MIMEImage(img_bytes.tobytes())
        img.add_header('Content-Disposition', 'attachment', filename="image.jpg")
        msg.attach(img)

        # Send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully!")

    # Save masks and tiles
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    output_tile_folder = os.path.join(output_folder, name + '_ForestTiles')

    if not os.path.exists(output_tile_folder):
        os.makedirs(output_tile_folder)

    smoke_path = os.path.join(output_tile_folder, 'Smoke_Mask.jpg')
    segmentation_mask.save(smoke_path)

    mask_path = os.path.join(output_tile_folder, 'Forest_Mask.jpg')
    mask_image = Image.new('L', image.size)
    mask_image.paste(ML_output)
    mask_image.save(mask_path)

    # Tile the mask image
    mask_image_array = np.array(mask_image)
    image_width, image_height = image.size
    tile_width, tile_height = 50, 50

    tile_array = []
    for i in range(0, image_width, tile_width):
        row = []
        for j in range(0, image_height, tile_height):
            right = min(i + tile_width, image_width)
            lower = min(j + tile_height, image_height)
            tile = mask_image.crop((i, j, right, lower))
            row.append(tile)
        tile_array.append(row)

    counter = 1
    for i in range(len(tile_array)):
        for j in range(len(tile_array[i])):
            sum_pixels = np.sum(np.array(tile_array[i][j]))
            max_sum = 255 * tile_width * tile_height
            if sum_pixels / max_sum > 0.70:
                right = min(i * tile_width + tile_width, image_width)
                lower = min(j * tile_height + tile_height, image_height)
                output_image_path = os.path.join(output_tile_folder, f'ForestTile{counter}.jpg')
                forest = image.crop((i * tile_width, j * tile_height, right, lower))
                forest.save(output_image_path)
                counter += 1

    print(f'Processed and saved forest tiles in folder: {output_tile_folder}')

input_folder = r'Input'
output_folder = r'Output'

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
