# HFest: HuggingFace Model Resource Estimator

A simple command line app that does estimation of the size of a Hugging Face model without downloading any of the models, saving precious time downloading large model files or quantizing and tweaking the model to fit in your system.

## Main Functionality
1. Find out whether a HuggingFace model is able to run on your system, particularly the GPU
2. How many resources does it need with several configurations
3. Suggest configuration for the most optimized settings

## Requirements
- Python 3.9 or newer

## Installation
1. Clone from the repository
```
git clone https://www.github.com/rezanatha/hfest.git
cd hfest
```
2. Install with [uv](https://docs.astral.sh/uv/) (Recommended)
```
uv run pip install -e .
```
3. Run and set your HuggingFace API
```
uv run hfest --help
uv run hfest config set api_key {YOUR_API_KEY}
```

4. Estimate model used storage
```
uv run hfest estimate-size {MODEL_ID}
```

5. Estimate model used storage and check whether the model(s) fit to your current GPU free memory
```
uv run hfest estimate-resource {MODEL_ID}
```