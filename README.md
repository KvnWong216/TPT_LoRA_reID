# TPT-LoRA-ReID
A robust Person Re-ID framework using LoRA (Low-Rank Adaptation) to combat illumination variations identified in TPT-Bench.

## Highlights
- **Backbone**: ResNet-50 (Frozen)
- **Adaptation**: LoRA injected into `conv2` layers.
- **Robustness**: Trained with extreme brightness augmentation (0.2x to 1.8x).

## Usage
1. `pip install -r requirements.txt`
2. `python main.py`