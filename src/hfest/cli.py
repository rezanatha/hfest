import argparse
import sys

from .commands import config, estimate_size, estimate_resource
from .version import __version__

def main():
    '''main entry'''

    parser = argparse.ArgumentParser(
        description="HFest: Hugging Face model size and resource estimator"
    )

    parser.add_argument('--version', action='version', 
                        version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # estimate-size
    estimate_size.setup_parser(subparsers)
    # estimate-resource
    estimate_resource.setup_parser(subparsers)  
    # config
    config.setup_parser(subparsers)

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        raise

    # If no command is specified, show help and exit
    if not args.command:
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    # Handle commands
    if args.command == "estimate-size":
        return estimate_size.handle(args)
    elif args.command == "estimate-resource":
        return estimate_resource.handle(args)
    elif args.command == "config":
        return config.handle(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())