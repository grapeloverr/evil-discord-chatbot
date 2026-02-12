#!/usr/bin/env bash
# OpenClaw Discord Bot Management Script

set -e

OPENCLAW_DIR="$HOME/.openclaw"
CONFIG_FILE="$OPENCLAW_DIR/openclaw.json"
TEMPLATE_FILE="$(dirname "$0")/openclaw.config.template.json"
ENV_FILE="$(dirname "$0")/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama not found. Install from: https://ollama.com"
        exit 1
    fi
    
    if ! command -v openclaw &> /dev/null && ! command -v node &> /dev/null; then
        log_error "OpenClaw or Node.js not found"
        exit 1
    fi
    
    log_success "Dependencies OK"
}

check_models() {
    log_info "Checking Ollama models..."
    
    local models=("kimi-k2.5:cloud" "qwen3-vl:235b-instruct-cloud")
    
    for model in "${models[@]}"; do
        if ollama list | grep -q "$model"; then
            log_success "Model available: $model"
        else
            log_warn "Model not found: $model"
            read -p "Pull $model? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                ollama pull "$model"
            fi
        fi
    done
}

setup_config() {
    log_info "Setting up OpenClaw configuration..."
    
    mkdir -p "$OPENCLAW_DIR"
    
    # Load .env if exists
    if [[ -f "$ENV_FILE" ]]; then
        log_info "Loading environment from .env"
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi
    
    # Check for Discord token
    if [[ -z "$DISCORD_BOT_TOKEN" ]]; then
        log_error "DISCORD_BOT_TOKEN not set!"
        echo ""
        echo "1. Go to: https://discord.com/developers/applications"
        echo "2. Create an application → Bot → Add Bot"
        echo "3. Enable Message Content Intent + Server Members Intent"
        echo "4. Copy the token and add to .env file:"
        echo ""
        echo "   DISCORD_BOT_TOKEN=your_token_here"
        echo ""
        exit 1
    fi
    
    # Create or update config
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_info "Creating new OpenClaw config..."
        cat > "$CONFIG_FILE" << EOF
{
  channels: {
    discord: {
      enabled: true,
      token: "$DISCORD_BOT_TOKEN",
      dm: {
        enabled: true,
        policy: "open",
        historyLimit: 30
      },
      groupPolicy: "open",
      historyLimit: 20,
      replyToMode: "first",
      mediaMaxMb: 25,
      allowBots: false
    }
  },
  agents: {
    defaults: {
      model: {
        primary: "ollama/kimi-k2.5:cloud",
        vision: "ollama/qwen3-vl:235b-instruct-cloud"
      },
      workspace: "$OPENCLAW_DIR/workspace"
    }
  },
  gateway: {
    port: 18789,
    bind: "loopback",
    mode: "local"
  },
  hooks: {
    internal: {
      enabled: true,
      entries: {
        "session-memory": {
          enabled: true
        }
      }
    }
  }
}
EOF
        log_success "Config created at $CONFIG_FILE"
    else
        log_warn "Config already exists at $CONFIG_FILE"
        log_info "Updating Discord token..."
        # Simple token update via node if available
        if command -v node &> /dev/null; then
            node -e "
const fs = require('fs');
const config = fs.readFileSync('$CONFIG_FILE', 'utf8');
// This is JSON5, so we do simple string replacement
const updated = config.replace(
  /token:\s*\"[^\"]*\"/,
  'token: \"$DISCORD_BOT_TOKEN\"'
);
fs.writeFileSync('$CONFIG_FILE', updated);
"
        fi
        log_success "Token updated"
    fi
}

start_gateway() {
    log_info "Starting OpenClaw gateway..."
    
    # Check if already running
    if curl -s http://127.0.0.1:18789/ > /dev/null 2>&1; then
        log_warn "Gateway already running on port 18789"
        read -p "Restart? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            openclaw gateway stop 2>/dev/null || true
            sleep 2
        else
            return
        fi
    fi
    
    openclaw gateway
}

stop_gateway() {
    log_info "Stopping OpenClaw gateway..."
    openclaw gateway stop 2>/dev/null || true
    log_success "Gateway stopped"
}

show_status() {
    log_info "OpenClaw Status:"
    echo ""
    
    if curl -s http://127.0.0.1:18789/ > /dev/null 2>&1; then
        log_success "Gateway: Running on port 18789"
        echo ""
        log_info "Dashboard: http://127.0.0.1:18789/"
    else
        log_warn "Gateway: Not running"
    fi
    
    echo ""
    log_info "Available Ollama models:"
    ollama list | head -10
    
    echo ""
    log_info "Config location: $CONFIG_FILE"
}

case "${1:-}" in
    setup)
        check_dependencies
        check_models
        setup_config
        ;;
    start)
        setup_config
        start_gateway
        ;;
    stop)
        stop_gateway
        ;;
    restart)
        stop_gateway
        sleep 2
        setup_config
        start_gateway
        ;;
    status)
        show_status
        ;;
    models)
        check_models
        ;;
    *)
        echo "OpenClaw Discord Bot Manager"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  setup     - Check dependencies and setup configuration"
        echo "  start     - Start the gateway (sets up config if needed)"
        echo "  stop      - Stop the gateway"
        echo "  restart   - Restart the gateway"
        echo "  status    - Show status of gateway and models"
        echo "  models    - Check and pull required Ollama models"
        echo ""
        echo "First time setup:"
        echo "  1. Copy .env.example to .env"
        echo "  2. Add your DISCORD_BOT_TOKEN to .env"
        echo "  3. Run: $0 setup"
        echo "  4. Run: $0 start"
        ;;
esac