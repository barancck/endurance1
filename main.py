import requests
#
# url = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
# headers = {
#     "Content-Type": "application/x-www-form-urlencoded"
# }
# data = {
#     "grant_type": "client_credentials",
#     "client_id": "5fe00af8-9737-45f6-9749-611869356681",
#     "client_secret": "uJoCDthcy6Lg04Eqv2ycMLoHagMowLQK"
# }
#
# response = requests.post(url, headers=headers, data=data)
#
# print(response.status_code)
# print(response.json())

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Your client credentials
client_id = "5fe00af8-9737-45f6-9749-611869356681"
client_secret = "uJoCDthcy6Lg04Eqv2ycMLoHagMowLQK"


def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)

print(token)
###


from sentinelhub import SHConfig

config = SHConfig()
config.sh_client_id = client_id
config.sh_client_secret = client_secret

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

betsiboka_coords_wgs84 = (46.16, -16.15, 46.51, -15.58)
resolution = 60
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {betsiboka_size} pixels")


evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2020-06-12", "2020-06-13"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config,
)

true_color_imgs = request_true_color.get_data()
print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
print(f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")




# curl -X POST --url https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token --header "content-type: application/x-www-form-urlencoded" --data "grant_type=client_credentials&client_id=5fe00af8-9737-45f6-9749-611869356681" --data-urlencode "client_secret=uJoCDthcy6Lg04Eqv2ycMLoHagMowLQK"