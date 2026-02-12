#!/bin/bash

# Discord Bot Quick Start Script
echo "🤖 Discord Bot Quick Start"
echo "========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Virtual environment created and dependencies installed"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please run ./setup_token.sh first to set up your bot token."
    exit 1
fi

# Check if token is configured
if grep -q "PASTE_NEW_TOKEN_HERE" .env; then
    echo "❌ Bot token not configured!"
    echo "Please run ./setup_token.sh to set up your bot token."
    exit 1
fi

# Test token first
echo "🧪 Testing Discord token..."
source venv/bin/activate && python -c "
import os
from dotenv import load_dotenv
import requests

load_dotenv()
token = os.getenv('DISCORD_BOT_TOKEN')

try:
    response = requests.get(
        'https://discord.com/api/v10/users/@me',
        headers={'Authorization': f'Bot {token}'},
        timeout=5
    )
    if response.status_code == 200:
        user_data = response.json()
        print('✅ Token valid - Bot: ', user_data.get('username', 'Unknown'))
    else:
        print(f'❌ Invalid token - Status: {response.status_code}')
        exit(1)
except Exception as e:
    print(f'❌ Token test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🚀 Starting bot..."
    echo "Press Ctrl+C to stop the bot"
    echo "============================"
    source venv/bin/activate && python bot.py
else
    echo "❌ Token validation failed!"
    echo "Please run ./setup_token.sh to fix your token."
fi