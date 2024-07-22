# import requests
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

# All requests using this session will have an access token automatically added
resp = oauth.get("https://services.sentinel-hub.com/configuration/v1/wms/instances")
print(resp.content)
