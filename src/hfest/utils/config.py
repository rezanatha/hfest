import json
import os

# Default config location
CONFIG_DIR = os.path.expanduser("~/.config/hfest")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# Default configuration
DEFAULT_CONFIG = {
    "default_model_path": None,
    "api_key": None,
}

def ensure_config_dir():
    """Ensure the config directory exists."""
    os.makedirs(CONFIG_DIR, exist_ok=True)

def save_config(config):
    """Save configuration to file."""
    ensure_config_dir()
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, indent=2, sort_keys=True, fp=f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def read_config():
    """Read configuration from file or create default if not exists."""
    ensure_config_dir()
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"Error reading config: {e}")
        return DEFAULT_CONFIG.copy()


def update_config(key, value):
    """Update a specific config value."""
    config = read_config()
    config[key] = value
    return save_config(config)
