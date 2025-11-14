
import argparse
from pydantic import BaseModel
from typing import Optional

from dmark.watermark.config import WatermarkConfig

class GenConfig(BaseModel):
    model: str
    dataset: str
    steps: int = 256
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"

class ExprConfig(BaseModel):
    num_samples: int
    output_dir: Optional[str]
    minimum_output_token: Optional[int]
    repeat_ratio: float = 0.2
    bitmap_device: str = "cpu"



def parse_args(): 
    parser = argparse.ArgumentParser()

    # we starts with dataset and model
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name or path",
    )

    # then we add number of samples and output directory
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--minimum_output_token", type=int, default=None)
    parser.add_argument("--repeat_ratio", type=float, default=0.2, 
                       help="Maximum ratio of any single token repetition (default: 0.2)")
    parser.add_argument("--output_dir", type=str, default=None)

    # then we add generation arguments
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random", "right_to_left", "left_to_right"])

    # now it's time to add watermark arguments 
    # even though many of them have default values, the watermark will only be enabled if the strategy is not None
    parser.add_argument("--strategy", type=str, default=None, choices=["normal", "predict", "bidirectional", "predict-bidirectional", "legacy-ahead", "legacy-both"])
    parser.add_argument("--bitmap", type=str, default="bitmap.bin")
    parser.add_argument("--bitmap_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to store the bitmap on")
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--prebias", type=bool, default=False)


    args = parser.parse_args()

    gen_config = GenConfig(
        model=args.model,
        dataset=args.dataset,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
    )

    watermark_config = WatermarkConfig(
        vocab_size=args.vocab_size,
        ratio=args.ratio,
        delta=args.delta,
        key=args.key,
        prebias=args.prebias,
        strategy=args.strategy,
        bitmap_path=args.bitmap
    )

    expr_config = ExprConfig(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        minimum_output_token=args.minimum_output_token,
        repeat_ratio=args.repeat_ratio,
        bitmap_device=args.bitmap_device,
    )

    return gen_config, watermark_config, expr_config

def generate_result_filename(
    gen_config: GenConfig,
    watermark_config: WatermarkConfig,
    expr_config: ExprConfig,
) -> str:
    """Generate a comprehensive result filename based on all configuration parameters."""
    import datetime
    
    # Extract model name (last part after /)
    model_name = gen_config.model.split('/')[-1]
    
    # Start with base components
    components = [
        "results",
        gen_config.dataset.replace('/', '_'),
        model_name,
    ]
    
    # Add generation parameters
    components.extend([
        f"steps{gen_config.steps}",
        f"len{gen_config.gen_length}",
        f"blk{gen_config.block_length}",
        f"temp{gen_config.temperature}",
        f"cfg{gen_config.cfg_scale}",
        f"mask_{gen_config.remasking}",
    ])
    
    # Add watermark parameters if enabled
    if watermark_config.strategy is not None:
        components.append("wm")
        components.extend([
            f"r{watermark_config.ratio}",
            f"d{watermark_config.delta}",
            f"k{watermark_config.key}",
            watermark_config.strategy,
        ])
        
        # Add prebias if enabled
        if watermark_config.prebias:
            components.append("prebias")
        
        # Add vocab size if not default
        if watermark_config.vocab_size != 126464:
            components.append(f"v{watermark_config.vocab_size}")
    else:
        components.append("nowm")
    
    # Add number of samples
    components.append(f"n{expr_config.num_samples}")
    
    # Add minimum output token if specified
    if expr_config.minimum_output_token is not None:
        components.append(f"min{expr_config.minimum_output_token}")
    
    # Add timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)
    
    # Join with underscores and add extension
    filename = "-".join(components) + ".json"
    
    return filename