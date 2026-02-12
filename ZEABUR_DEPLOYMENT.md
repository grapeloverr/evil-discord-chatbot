# Deploy to Zeabur

## Environment Variables

When deploying to Zeabur, add these variables in the "Variables" tab:

### Required
| Key | Value | Description |
|-----|-------|-------------|
| `DISCORD_BOT_TOKEN` | `YOUR_TOKEN_HERE` | Your Discord bot token |
| `ZEABUR_ENVIRONMENT` | `production` | Enables the web server for health checks |

### Optional
| Key | Value | Description |
|-----|-------|-------------|
| `PIN_RESEND_CHANNEL_ID` | `Your_Channel_ID` | Channel ID to resend pins to (per-server config preferred) |
| `MAX_CHANNEL_MESSAGES` | `30` | Max messages per channel (default: 30) |
| `AUTO_DELETE_ENABLED` | `true` | Enable auto-deletion (default: false) |
| `FILTER_ENABLED` | `false` | Enable/disable word filter (default: false) |

## Cleanup
You can remove the "PASSWORD" and "PORT" variables if they were auto-generated and not used by your other services, though `PORT` is usually automatically managed by Zeabur and can be left alone.
