# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from starlette.responses import HTMLResponse

from dotenv import load_dotenv

load_dotenv()

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

TO_NUMBER = "+91-----------"  #  number

call = client.calls.create(
    twiml=open("templates/streams.xml").read(),
    to=TO_NUMBER,
    from_=os.environ["TWILIO_PHONE_NUMBER"]
)

print(call.sid)