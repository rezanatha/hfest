def setup_parser(subparsers):
    parser = subparsers.add_parser("estimate-resource", help = "Estimate model size and resource needed to run the model")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b)")
    parser.add_argument("gpu_config", help="GPU config the model is running on (0: auto, 1: single setup, 2: distributed setup, )")
    parser.add_argument("quantization", help="level of quantization (0: auto, 1: minimal, 2: adequate, 3: aggresive)")
    return parser

def handle(args):
    print(f"Estimating resources for model: {args.model_id}")
    print("handle_estimate_resource not implemented")
    # estimate model size
    # detect host GPU specifications (from something like nvidia-smi)
    # compare gpu spec with model size, is it possible to run on it?
    # how many resources would it take?
    # if not quantized/quantized (4FP, 16FP, etc)
    # if running on a single/distributed system
    # provide recommendations
    return 0   