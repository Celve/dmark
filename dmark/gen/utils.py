import argparse
import datetime
from typing import Optional

from dmark.watermark.pattern_mark import PatternMark
from pydantic import BaseModel

from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.base import BaseWatermark
from dmark.watermark.bidirectional import BidirectionalWatermark
from dmark.watermark.kgw import KGWWatermark
from dmark.watermark.predictive import PredictiveWatermark
from dmark.watermark.predictive_bidirectional import PredictiveBidirectionalWatermark


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
    batch_size: int = 1
    bitmap_device: str = "cpu"
    ignore_eos: bool = False


class DreamGenConfig(BaseModel):
    model: str
    dataset: str
    steps: int = 256
    gen_length: int = 256
    temperature: float = 0.2
    alg: str = "entropy"
    alg_temp: float = 0.0
    eps: float = 1e-3
    top_p: float = 0.95
    top_k: Optional[int] = None
    output_history: bool = False
    return_dict_in_generate: bool = True


class DreamExprConfig(BaseModel):
    num_samples: int
    output_dir: Optional[str]
    minimum_output_token: Optional[int]
    repeat_ratio: float = 0.2
    batch_size: int = 1
    bitmap_device: str = "cpu"
    ignore_eos: bool = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name or path",
    )

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Prompts per forward pass"
    )
    parser.add_argument("--minimum_output_token", type=int, default=None)
    parser.add_argument(
        "--repeat_ratio",
        type=float,
        default=0.2,
        help="Maximum ratio of any single token repetition (default: 0.2)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Prevent generation from sampling the EOS token",
    )

    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random", "right_to_left", "left_to_right"],
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=[
            "normal",
            "predict",
            "bidirectional",
            "predict-bidirectional",
            "pattern-mark",
        ],
    )
    parser.add_argument("--bitmap", type=str, default="bitmap.bin")
    parser.add_argument(
        "--bitmap_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to store the bitmap on",
    )
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument(
        "--pattern_length",
        type=int,
        default=8,
        help="Pattern length for pattern-mark watermarking",
    )

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

    watermark_config: dict[str, object] = {
        "vocab_size": args.vocab_size,
        "ratio": args.ratio,
        "delta": args.delta,
        "key": args.key,
        "strategy": args.strategy,
        "bitmap_path": args.bitmap,
        "pattern_length": args.pattern_length,
    }

    expr_config = ExprConfig(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        minimum_output_token=args.minimum_output_token,
        repeat_ratio=args.repeat_ratio,
        batch_size=args.batch_size,
        bitmap_device=args.bitmap_device,
        ignore_eos=args.ignore_eos,
    )

    return gen_config, watermark_config, expr_config


def parse_dream_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="Dream-org/Dream-v0-Instruct-7B",
        help="Dream model name or path",
    )

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1, help="Prompts per batch")
    parser.add_argument("--minimum_output_token", type=int, default=None)
    parser.add_argument(
        "--repeat_ratio",
        type=float,
        default=0.2,
        help="Maximum ratio of any single token repetition (default: 0.2)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Prevent generation from sampling the EOS token",
    )

    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--alg",
        type=str,
        default="entropy",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
    )
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--output_history",
        action="store_true",
        help="Return the diffusion history from DREAM",
    )
    return_dict_group = parser.add_mutually_exclusive_group()
    return_dict_group.add_argument(
        "--return_dict_in_generate",
        action="store_true",
        dest="return_dict_in_generate",
        help="Return DreamModelOutput from diffusion_generate",
    )
    return_dict_group.add_argument(
        "--no-return_dict_in_generate",
        action="store_false",
        dest="return_dict_in_generate",
        help="Return tensor output only",
    )
    parser.set_defaults(return_dict_in_generate=True)

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["normal", "predict", "bidirectional", "predict-bidirectional", "pattern-mark"],
    )
    parser.add_argument("--bitmap", type=str, default="bitmap.bin")
    parser.add_argument(
        "--bitmap_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to store the bitmap on",
    )
    parser.add_argument("--vocab_size", type=int, default=152064)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument(
        "--pattern_length",
        type=int,
        default=8,
        help="Pattern length for pattern-mark watermarking",
    )

    args = parser.parse_args()

    gen_config = DreamGenConfig(
        model=args.model,
        dataset=args.dataset,
        steps=args.steps,
        gen_length=args.gen_length,
        temperature=args.temperature,
        alg=args.alg,
        alg_temp=args.alg_temp,
        eps=args.eps,
        top_p=args.top_p,
        top_k=args.top_k,
        output_history=args.output_history,
        return_dict_in_generate=args.return_dict_in_generate,
    )

    watermark_config: dict[str, object] = {
        "vocab_size": args.vocab_size,
        "ratio": args.ratio,
        "delta": args.delta,
        "key": args.key,
        "strategy": args.strategy,
        "bitmap_path": args.bitmap,
        "pattern_length": args.pattern_length,
    }

    expr_config = DreamExprConfig(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        minimum_output_token=args.minimum_output_token,
        repeat_ratio=args.repeat_ratio,
        batch_size=args.batch_size,
        bitmap_device=args.bitmap_device,
        ignore_eos=args.ignore_eos,
    )

    return gen_config, watermark_config, expr_config


def _finalize_filename_components(
    base_components: list[str],
    watermark_config: dict[str, object],
    expr_config: BaseModel,
    vocab_default: int,
) -> str:
    components = list(base_components)
    strategy = watermark_config.get("strategy")
    if strategy is not None:
        components.append("wm")
        components.extend(
            [
                f"r{watermark_config.get('ratio')}",
                f"d{watermark_config.get('delta')}",
                f"k{watermark_config.get('key')}",
                strategy,
            ]
        )
        vocab_size = watermark_config.get("vocab_size", vocab_default)
        if vocab_size != vocab_default:
            components.append(f"v{vocab_size}")
    else:
        components.append("nowm")

    num_samples = getattr(expr_config, "num_samples", None)
    if num_samples is not None:
        components.append(f"n{num_samples}")
    minimum_output_token = getattr(expr_config, "minimum_output_token", None)
    if minimum_output_token is not None:
        components.append(f"min{minimum_output_token}")
    if getattr(expr_config, "ignore_eos", False):
        components.append("noeos")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)
    return "-".join(components) + ".json"


def generate_result_filename(
    gen_config: GenConfig,
    watermark_config: dict[str, object],
    expr_config: ExprConfig,
) -> str:
    model_name = gen_config.model.split("/")[-1]
    base = [
        "results",
        gen_config.dataset.replace("/", "_"),
        model_name,
        f"steps{gen_config.steps}",
        f"len{gen_config.gen_length}",
        f"blk{gen_config.block_length}",
        f"temp{gen_config.temperature}",
        f"cfg{gen_config.cfg_scale}",
        f"mask_{gen_config.remasking}",
    ]
    return _finalize_filename_components(base, watermark_config, expr_config, 126464)


def generate_dream_result_filename(
    gen_config: DreamGenConfig,
    watermark_config: dict[str, object],
    expr_config: DreamExprConfig,
) -> str:
    model_name = gen_config.model.split("/")[-1]
    base = [
        "results",
        gen_config.dataset.replace("/", "_"),
        model_name,
        "dream",
        f"steps{gen_config.steps}",
        f"len{gen_config.gen_length}",
        f"temp{gen_config.temperature}",
        f"alg_{gen_config.alg}",
        f"alg_temp{gen_config.alg_temp}",
    ]
    if gen_config.eps != 1e-3:
        base.append(f"eps{gen_config.eps}")
    if gen_config.top_p != 0.95:
        base.append(f"top_p{gen_config.top_p}")
    if gen_config.top_k is not None:
        base.append(f"top_k{gen_config.top_k}")
    return _finalize_filename_components(base, watermark_config, expr_config, 152064)


_WATERMARK_CLASS_MAP: dict[str, type[BaseWatermark]] = {
    "normal": KGWWatermark,
    "predict": PredictiveWatermark,
    "bidirectional": BidirectionalWatermark,
    "predict-bidirectional": PredictiveBidirectionalWatermark,
}


def build_watermark(
    watermark_config: dict[str, object],
    *,
    bitmap_device: str,
    mask_id: int,
) -> BaseWatermark | None:
    """Instantiate the requested watermark class for generation pipelines.

    Args:
        watermark_config: User provided watermark configuration as a plain dict.
        bitmap_device: Device where the persistent bitmap should be kept.
        mask_id: Mask token id used by the generator (needed for BaseWatermark).

    Returns:
        A concrete ``BaseWatermark`` implementation or ``None`` when watermarking
        is disabled.
    """

    strategy = watermark_config.get("strategy")
    if strategy is None:
        return None

    if strategy == "pattern-mark":
        return PatternMark(watermark_config, mask_id)

    watermark_cls = _WATERMARK_CLASS_MAP.get(strategy)
    if watermark_cls is None:
        raise ValueError(f"Unsupported watermark strategy '{strategy}'.")

    bitmap = PersistentBitmap(
        watermark_config["vocab_size"],
        watermark_config["bitmap_path"],
        device=bitmap_device,
    )
    return watermark_cls(watermark_config, bitmap, mask_id)
