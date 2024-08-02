
import os
import pandas as pd
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, bbox_to_dimensions, BBox, CRS, MimeType
from PIL import Image
import numpy as np
from datetime import datetime, timedelta  # Import datetime library


# Ensure the images folder exists
os.makedirs('images3', exist_ok=True)

# Step 1: Read the CSV file
csv_file_path = 'italia.csv'
data = pd.read_csv(csv_file_path)

# Step 2: Process the parameters
parameters = data.to_dict('records')

# Configure Sentinel Hub credentials
config = SHConfig()
config.sh_client_id = '5fe00af8-9737-45f6-9749-611869356681'  # Replace with your actual client ID
config.sh_client_secret = 'uJoCDthcy6Lg04Eqv2ycMLoHagMowLQK'  # Replace with your actual client secret


def download_images(parameters, config):
    counter = 0
    for param in parameters:
        latitude = param['latitude']
        longitude = param['longitude']
        date = param['acq_date']

        # Parse the date string into a datetime object
        start_date = datetime.strptime(date, '%m/%d/%Y')
        end_date = start_date + timedelta(days=-800)
        extra_date = start_date + timedelta(days=-365)
        # Set to one day later

        # Define bounding box
        bbox = BBox(bbox=[longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01], crs=CRS.WGS84)
        dimensions = bbox_to_dimensions(bbox, resolution=10)  # Update resolution as needed

        # Define the Sentinel Hub request
        request = SentinelHubRequest(
            evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["B08", "B11", "B12", "B04", "SCL"],
                            output: { bands: 4 }
                        };
                    }
                    function evaluatePixel(sample) {
                        return [sample.B08, sample.B11, sample.B12, sample.B04];
                    }
                    """,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A,
                            time_interval=(end_date, extra_date)
                        )
                    ],
                    bbox=bbox,
                    size=(224, 224),
                    config=config,
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)]
                )

        # Execute the request and save the output
        try:
            response = request.get_data()
            if not response or len(response) == 0:
                print(f"No data returned for coordinates ({latitude}, {longitude}) on {date}.")
                continue

            # Save the image using PIL
            for i, img in enumerate(response):
                if img is None or img.size == 0:
                    print(f"No image data received for coordinates ({latitude}, {longitude}) on {date}.")
                    continue

                # Normalize the image to 8-bit
                img = (img - img.min()) / (img.max() - img.min()) * 255
                img = img.astype(np.uint8)

                image = Image.fromarray(img)
                print(counter)
                image_file_path = os.path.join('images3', f'italynofire{counter}.tiff')
                image.save(image_file_path)
                print(f"Saved image to {image_file_path}")
            counter += 1
            print(counter)
        except Exception as e:
            print(f"An error occurred while downloading or saving image for ({latitude}, {longitude}) on {date}: {e}")


# Call the function to download images
download_images(parameters, config)


#quick sidenote, you can add this to sort through clouds but its kinda pointless if you have a cloud filtering script
# evalscript = """
# //VERSION=3
#
# function setup() {
#     return {
#         input: ["B08", "B11", "B12", "B04", "SCL"],
#         output: { bands: 4 }
#     };
# }
#
# function evaluatePixel(sample) {
#     function isCloudy(cloudProbability) {
#         return cloudProbability > 60;
#     }
#
#     var cloudProbability = (sample.SCL === 9) ? 100 : 0;
#
#     if (isCloudy(cloudProbability)) {
#         return [0, 0, 0, 0];
#     } else {
#         return [sample.B08, sample.B11, sample.B12, sample.B04];
#     }
# }
# """
#
# input_data = [
#     SentinelHubRequest.input_data(
#         data_collection=DataCollection.SENTINEL2_L2A,
#         time_interval=(end_date, extra_date)
#     )
# ]
#
# bbox = bbox
# size = (224, 224)
# config = config
#
# request = SentinelHubRequest(
#     evalscript=evalscript,
#     input_data=input_data,
#     bbox=bbox,
#     size=size,
#     config=config,
#     responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)]
# )
