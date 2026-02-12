# Discord AI Bot v2.0

A modern, modular Discord bot with AI capabilities, memory, and optional voice features.

## Features

- **🤖 AI Chat**: Powered by Ollama or OpenAI models
- **💾 Memory**: Persistent conversation history with SQLite
- **🔧 Modular**: Plugin-based architecture for easy extension
- **🎙️ Voice**: Optional voice chat with speech recognition and TTS
- **⚙️ Configurable**: YAML-based configuration with environment overrides
- **🐳 Docker**: Containerized deployment support

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Discord bot token (get from [Discord Developer Portal](https://discord.com/developers/applications))
- Ollama running locally (or OpenAI API key)

### 2. Setup

```bash
# Clone and navigate
cd discord-bot

# Copy environment file
cp .env.example .env

# Edit .env with your Discord token
nano .env
```

### 3. Install Dependencies

```bash
# Core dependencies only
pip install -r requirements.txt

# Or with voice features (optional)
pip install -r requirements.txt
# Uncomment voice dependencies in requirements.txt first
```

### 4. Run

```bash
# Development mode
python main.py

# Or with Docker
docker-compose up
```

## Configuration

### Environment Variables

Key variables in `.env`:

```bash
DISCORD_TOKEN=your_discord_bot_token_here
BOT_ENV=production
LOCAL_LLM_URL=http://127.0.0.1:11434
MODEL_NAME=dolphin-llama3
```

### YAML Configuration

Edit `config/settings.yaml` for advanced configuration:

```yaml
# Discord settings
discord:
  prefix: "!"
  mention_prefix: true
  intents: [messages, guilds, message_content]

# LLM settings
llm:
  provider: "ollama"  # or "openai"
  model: "dolphin-llama3"
  temperature: 0.8
  max_tokens: 150

# Memory settings
memory:
  type: "sqlite"
  max_conversations: 1000
  persistence: true

# Voice settings (optional)
voice:
  enabled: false
  speech_recognition: true
  text_to_speech: true
```

## Commands

| Command | Description |
|---------|-------------|
| `!chat <message>` | Chat with the AI |
| `!model [name]` | Show or switch AI models |
| `!status` | Show bot status |
| `!remember key := value` | Learn a new fact |
| `!recall <query>` | Search memories |
| `!help` | Show help information |
| `!join` | Join voice channel |
| `!leave` | Leave voice channel |
| `!voice` | Show voice status |

## Voice Features (Optional)

Enable voice features by:

1. Install voice dependencies:
   ```bash
   pip install discord-ext-voice-recv SpeechRecognition edge-tts
   ```

2. Enable in configuration:
   ```yaml
   voice:
     enabled: true
   ```

3. Use `!join` in a voice channel to start voice chat

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start with Ollama
docker-compose up

# Or bot only (external Ollama)
docker-compose up discord-bot
```

### Manual Docker

```bash
# Build image
docker build -t discord-bot .

# Run container
docker run -d \\
  --name discord-bot \\
  -e DISCORD_TOKEN=your_token \\
  -v $(pwd)/data:/app/data \\
  discord-bot
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/

# Type checking
mypy src/
```

### Project Structure

```
discord-bot/
├── src/
│   ├── bot/           # Core bot functionality
│   ├── plugins/       # Optional plugins
│   └── utils/         # Utilities
├── config/            # Configuration files
├── tests/             # Test suite
├── docker/            # Docker files
└── main.py           # Entry point
```

## LLM Providers

### Ollama (Default)

1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull dolphin-llama3`
3. Set `LOCAL_LLM_URL=http://127.0.0.1:11434`

### OpenAI

1. Set `OPENAI_API_KEY` in environment
2. Configure provider in `settings.yaml`:
   ```yaml
   llm:
     provider: "openai"
     model: "gpt-3.5-turbo"
   ```

## Troubleshooting

### Bot Won't Start

- Check `DISCORD_TOKEN` is set correctly
- Verify bot has proper intents enabled
- Check logs for error messages

### Memory Issues

- Ensure SQLite database is writable
- Check `memory.persistence` setting
- Monitor database size with `!status`

### Voice Problems

- Install voice dependencies
- Check bot has voice channel permissions
- Verify `voice.enabled: true` in config

### LLM Connection Issues

- Check Ollama is running: `ollama list`
- Verify `LOCAL_LLM_URL` is correct
- Test with `!status` command

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
