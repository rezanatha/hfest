from utils.config import read_config, update_config


def setup_parser(subparsers):
    parser = subparsers.add_parser("config", help="Manage configuration")

    # Add subcommands for config: get, set, list
    config_subparsers = parser.add_subparsers(dest="config_command", help="Config commands")
    
    # Get a specific config value
    get_parser = config_subparsers.add_parser("get", help="Get a config value")
    get_parser.add_argument("key", help="Config key to retrieve")
    
    # Set a config value
    set_parser = config_subparsers.add_parser("set", help="Set a config value")
    set_parser.add_argument("key", help="Config key to set")
    set_parser.add_argument("value", help="Value to set")
    
    # List all config values
    list_parser = config_subparsers.add_parser("list", help="List all config values")
    
    return parser


def handle(args):
    '''handle the config command'''
    if args.config_command == "get":
        return handle_get(args)
    
    elif args.config_command == "set":
        return handle_set(args)
    
    elif args.config_command == "list":
        return handle_list(args)
    
    else:
        print("Please specify a config subcommand. Use --help for more information.")
        return 1
    
def handle_get(args):
    config = read_config()

    if args.key in config:
        print(f"{args.key} : {config[args.key]}")
        return 0

    else:
        print(f"Config key {args.key} not found.")
        return 1
    

def handle_set(args):
    config = read_config()
    if args.key not in config:
        print(f"Unknown config key: {args.key}")
        return 1
    result = update_config(args.key, args.value)
    if result:
        print(f"Updated {args.key}: {args.value}")
        return 0
    else:
        print(f"Failed to update {args.key}: {args.value}")
        return 1

def handle_list(args):
    config = read_config()
    print("Current config")
    for k, v in config.items():
        print(f"{k}: {v}")
    return 0