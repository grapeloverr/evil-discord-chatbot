#!/usr/bin/env python3
"""
Zeabur deployment webhook for Discord bot
"""

import os
import sys
from flask import Flask, jsonify
from bot import WebhookBot

app = Flask(__name__)


@app.route("/health")
def health_check():
    """Health check endpoint for Zeabur"""
    return jsonify({"status": "healthy", "service": "discord-webhook-bot"})


@app.route("/")
def index():
    """Basic index route"""
    return jsonify({"message": "Discord Webhook Bot is running!"})


if __name__ == "__main__":
    # Check if we're in a production environment (Zeabur)
    if os.getenv("ZEABUR_ENVIRONMENT"):
        # Zeabur deployment - run web server
        port = int(os.getenv("PORT", 8080))

        # Start Discord bot in background
        import threading

        def run_bot():
            token = os.getenv("DISCORD_BOT_TOKEN")
            if not token:
                print("Error: DISCORD_BOT_TOKEN environment variable not set")
                return
            bot = WebhookBot()
            bot.run(token)

        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()

        # Run Flask app
        app.run(host="0.0.0.0", port=port)
    else:
        # Local development - run bot directly
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            print("Error: DISCORD_BOT_TOKEN environment variable not set")
            sys.exit(1)
        bot = WebhookBot()
        bot.run(token)
