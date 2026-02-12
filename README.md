# Discord Webhook Bot

A production-ready Discord bot that implements a `/msg` slash command for sending messages via webhooks while perfectly impersonating the user who runs the command.

## Features

- **User Impersonation**: Messages appear exactly as if sent by user who runs the command
- **All Attachment Support**: Support for images, videos, audio, PDF, and text files up to 8MB
- **Channel Cleanup**: Automatically maintains maximum of 30 messages per channel (excluding pins)
- **Pin Resending**: Pinned messages are automatically resent to specified channel with all attachments
- **Command Deletion**: Original `/msg` command invocation is automatically deleted
- **Webhook Caching**: Efficient webhook management with TTL-based caching
- **Moderation Tools**: Comprehensive ban, kick, mute, timeout, and clear commands
- **Permission Safety**: Role hierarchy checks and permission validation
- **Comprehensive Error Handling**: Graceful handling of permissions, API limits, and edge cases
- **Production Ready**: Full logging, systemd support, and CLI interface
- **Cross-Platform**: Supports Linux (CachyOS/Arch) and Windows

## Quick Start

### 1. Installation

```bash
# Clone or download the bot files
cd webhook_obsecure

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your bot token
```

### 2. Bot Setup

1. **Create Bot Application**:
   - Visit [Discord Developer Portal](https://discord.com/developers/applications)
   - Create new application → Bot → Add Bot
   - Enable **Message Content Intent**
   - Copy bot token to `.env` file

2. **Invite Bot to Server**:
   - Generate OAuth2 URL with these permissions:
     - Manage Webhooks
     - Send Messages
     - Read Message History
     - Embed Links
     - Attach Files
     - Manage Messages
     - Read Messages
   - Use URL format: `https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=68608&scope=bot%20applications.commands`

3. **Run Bot**:
   ```bash
   python bot.py
   ```

## Usage

### Slash Command

```
/msg Your message here
```

**With attachment:**
```
/msg Your message here attachment: [upload any file]
```

**Manual pin resending:**
```
/resend_pin message_id:123456789
```

**Channel management:**
```
/autodelete enabled:true    # Enable 30 message limit
/autodelete enabled:false   # Disable automatic cleanup
```

**Moderation commands:**
```
/ban user:@User reason:"Breaking rules" delete_message_days:7
/kick user:@User reason:"Warning"
/mute user:@User duration:60 reason:"Spam"
/unmute user:@User reason:"Appealed"
/timeout user:@User duration:120 reason:"Toxic behavior"
/clear amount:50 user:@User
/lq user:@User
```

**Prefix commands (with CUSTOM_PREFIX="~"):**
```
~lq @User    # Punish user (remove roles, add punish role)
~ulq @User   # Unpunish user (restore previous roles)
```

### Automatic Features

**Channel Cleanup:**
- Bot maintains configurable message limit per channel (when enabled)
- Must be enabled per-server using `/autodelete enabled:true`
- Default limit is 30 messages, configurable via `AUTO_DELETE_COUNT`
- Set to 5 to delete every 5th message, 10 for every 10th, etc.
- Pinned messages are excluded from cleanup
- Oldest messages are deleted first when limit is exceeded
- Bot messages are ignored in cleanup
- Requires 'Manage Messages' permission to use the command

**Moderation Features:**
- **Ban**: Permanently remove users with optional message cleanup
- **Kick**: Remove users temporarily with reason logging
- **Mute/Unmute**: Control user speaking permissions with duration
- **Timeout**: Temporary user silencing with custom duration
- **Clear**: Bulk message deletion with user-specific filtering
- **Punishment System**: Role-based punishment with role restoration
- **Permission checks**: Role hierarchy and permission validation
- **Audit logging**: All moderation actions logged with details

**Custom Prefix Support:**
- **Configurable prefix**: Set `CUSTOM_PREFIX="~"` for ~lq commands
- **Dual interface**: Both slash commands (/lq) and prefix commands (~lq)
- **Role management**: Store and restore user roles during punishment
- **Smart restoration**: Automatically restore previous roles on unpunish

**Persistent Storage:**
- **Auto-delete settings**: Remember which servers have auto-delete enabled
- **Punishment roles**: Store user roles across bot restarts
- **Pin tracking**: Prevent duplicate resends after reconnection
- **JSON storage**: All settings saved in local JSON files
- **Automatic cleanup**: Periodic data maintenance and cleanup

**Pin Resending:**
- When a message is pinned, it's automatically resent to the channel specified in `PIN_RESEND_CHANNEL_ID`
- Original author identity is preserved via webhook impersonation
- **All attachments included** - images, videos, audio, PDF, text files (up to 8MB each)
- Works even if original message is deleted from source channel
- Manual trigger available with `/resend_pin` command
- Detects pin events through message edits and reactions

The bot will:
1. Create/retrieve a webhook in the channel
2. Send your message via webhook (appears as if sent by you)
3. Include any attachments with proper validation
4. Delete the original command invocation
5. Monitor channel for cleanup and pin management

**Supported File Types:**
- **Images**: PNG, JPEG, GIF, WebP
- **Videos**: MP4, MOV, AVI, WebM, etc.
- **Audio**: MP3, WAV, OGG, M4A, etc.
- **Documents**: PDF, TXT, and other text files
- **Maximum file size**: 8MB per file

### CLI Options

```bash
# Show setup instructions
python bot.py --setup

# Check configuration
python bot.py --config-check

# Run bot normally
python bot.py
```

## Technical Details

### Architecture

- **Time Complexity**: O(1) for webhook cache hits, O(n) for cache misses (n = webhooks in channel)
- **Space Complexity**: O(m) where m = number of cached webhooks
- **Cache TTL**: 1 hour with automatic expiration
- **Thread Safety**: Asyncio locks for concurrent access

### Security Features

- Environment variable configuration (no hardcoded tokens)
- Permission validation before operations
- Rate limit awareness and retry logic
- Comprehensive logging for monitoring

### Error Handling

- Missing permissions (Manage Webhooks, Send Messages, Delete Messages, Attach Files)
- Invalid message lengths (Discord's 2000 character limit)
- Invalid image formats or oversized files (8MB limit)
- Non-text channel validation
- Webhook creation failures
- Network timeouts and API errors

## Deployment

### Systemd Service (Linux)

```bash
# Create user
sudo useradd -r -s /bin/false discord

# Deploy files
sudo mkdir -p /opt/webhook-bot
sudo cp * /opt/webhook-bot/
sudo chown -R discord:discord /opt/webhook-bot

# Install service
sudo cp webhook-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable webhook-bot
sudo systemctl start webhook-bot

# Check status
sudo systemctl status webhook-bot
sudo journalctl -u webhook-bot -f
```

### Windows Service

Use NSSM (Non-Sucking Service Manager):
```cmd
nssm install WebhookBot python "C:\path\to\bot.py"
nssm set WebhookBot AppDirectory "C:\path\to\webhook_obsecure"
nssm start WebhookBot
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|-----------|-------------|
| `DISCORD_BOT_TOKEN` | Yes | Bot token from Discord Developer Portal |
| `DEBUG_GUILD_ID` | No | Guild ID for testing commands (faster sync) |
| `PIN_RESEND_CHANNEL_ID` | No | Channel ID where pinned messages are resent |
| `AUTO_DELETE_COUNT` | No | Maximum messages to keep in channel (default: 30) |
| `CUSTOM_PREFIX` | No | Custom command prefix (default: "!") |
| `PUNISH_ROLE_ID` | No | Role ID for punishment system |

### Logging

- File: `bot.log` (in bot directory)
- Console: Real-time output

## Troubleshooting

### Common Issues

1. **"Missing permissions" error**:
   - Ensure bot has Manage Webhooks permission
   - Check channel-specific overrides
   - Verify Attach Files permission for image support

2. **"Message content intent" error**:
   - Enable Message Content Intent in Developer Portal
   - Re-invite bot to server with new permissions

3. **Commands not appearing**:
   - Wait up to 1 hour for global sync
   - Use DEBUG_GUILD_ID for instant sync in test server

4. **Webhook creation fails**:
   - Check bot has Manage Webhooks permission
   - Verify channel allows webhooks

5. **Image upload fails**:
   - Verify image format (PNG, JPEG, GIF, WebP)
   - Check file size (max 8MB)
   - Ensure Attach Files permission

### Debug Mode

Set `DEBUG_GUILD_ID` in `.env` to test commands in a specific server without waiting for global sync.

## Development

### Project Structure

```
webhook_obsecure/
├── bot.py              # Main bot implementation
├── requirements.txt    # Python dependencies
├── .env.example       # Environment template
├── webhook-bot.service # Systemd service file
└── README.md          # This documentation
```

### Dependencies

- `discord.py>=2.3.0` - Discord API wrapper
- `python-dotenv>=1.0.0` - Environment variable management

## License

This project is provided as-is for educational and personal use.
