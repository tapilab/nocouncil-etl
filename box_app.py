"""
Authenticate with box to get refresh and access tokens, written to .env.
"""
import os
from flask import Flask, request, redirect
from dotenv import load_dotenv
import requests

load_dotenv()
app = Flask(__name__)

CLIENT_ID     = os.getenv("BOX_CLIENT_ID")
CLIENT_SECRET = os.getenv("BOX_CLIENT_SECRET")
REDIRECT_URI  = "http://127.0.0.1:5001/callback"

AUTH_URL  = "https://account.box.com/api/oauth2/authorize"
TOKEN_URL = "https://api.box.com/oauth2/token"

@app.route('/')
def index():
    # Step 1: redirect user to Box’s consent page
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
    }
    url = AUTH_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return redirect(url)

@app.route('/callback')
def callback():
    # Box redirects here with ?code=AUTH_CODE
    code = request.args.get("code")
    data = {
        "grant_type":    "authorization_code",
        "code":          code,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
    }
    # Step 2: exchange for tokens
    resp = requests.post(TOKEN_URL, data=data).json()
    access_token  = resp["access_token"]
    refresh_token = resp["refresh_token"]

    # Step 3: Persist tokens back into .env
    with open(".env", "a") as f:
        f.write(f"\nBOX_ACCESS_TOKEN={access_token}")
        f.write(f"\nBOX_REFRESH_TOKEN={refresh_token}")

    return "✅ Tokens saved to .env. You can now stop this server."
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)