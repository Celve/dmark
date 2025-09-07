import argparse
import json
import os

from pydantic import BaseModel
from typing import Any, Optional

import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.watermark import Watermark

device = "cuda"

class GenConfig(BaseModel):
    model: str
    dataset: str
    steps: int = 512
    gen_length: int = 256
    temperature: float = 0.2
    # DREAM-specific parameters
    alg: str = "entropy"  # "origin", "maskgit_plus", "topk_margin", "entropy"
    alg_temp: float = 0.0
    eps: float = 1e-3
    top_p: float = 0.95
    top_k: Optional[int] = None
    output_history: bool = False
    return_dict_in_generate: bool = True

class ExprConfig(BaseModel):
    num_samples: int
    output_dir: Optional[str]
    minimum_output_token: Optional[int]

def parse_args(): 
    parser = argparse.ArgumentParser()

    # we starts with dataset and model
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="Dream-org/Dream-v0-Instruct-7B",
        help="DREAM model name or path",
    )

    # then we add number of samples and output directory
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--minimum_output_token", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # then we add DREAM generation arguments
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--alg", type=str, default="entropy", choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--output_history", action="store_true", help="Output generation history")
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True, help="Return dict in generate")

    # now it's time to add watermark arguments 
    # even though many of them have default values, the watermark will only be enabled if the strategy is not None
    parser.add_argument("--strategy", type=str, default=None, choices=["normal", "predict", "reverse"])
    parser.add_argument("--bitmap", type=str, default="bitmap.bin")
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
        temperature=args.temperature,
        alg=args.alg,
        alg_temp=args.alg_temp,
        eps=args.eps,
        top_p=args.top_p,
        top_k=args.top_k,
        output_history=args.output_history,
        return_dict_in_generate=args.return_dict_in_generate,
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
        "dream",  # Indicate this is DREAM model
    ]
    
    # Add generation parameters
    components.extend([
        f"steps{gen_config.steps}",
        f"len{gen_config.gen_length}",
        f"temp{gen_config.temperature}",
        f"alg_{gen_config.alg}",
        f"alg_temp{gen_config.alg_temp}",
    ])
    
    # Add optional DREAM parameters
    if gen_config.eps != 1e-3:
        components.append(f"eps{gen_config.eps}")
    if gen_config.top_p != 0.95:
        components.append(f"top_p{gen_config.top_p}")
    if gen_config.top_k is not None:
        components.append(f"top_k{gen_config.top_k}")
    
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


def run_generation(
    gen_config: GenConfig,
    watermark_config: WatermarkConfig,
    expr_config: ExprConfig,
) -> list[dict[str, Any]]:
    """Run the generation process with optional watermarking."""
    # Load dataset - handle different dataset configurations
    if gen_config.dataset == "allenai/c4":
        # Load C4 dataset with streaming for memory efficiency
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        dataset_iter = iter(dataset)
        dataset_format = "text"  # C4 has 'text' field
        use_streaming = True
    elif gen_config.dataset == "openai/openai_humaneval":
        # Load HumanEval dataset (only has test split)
        dataset = load_dataset("openai/openai_humaneval", split="test")
        dataset_format = "code"  # HumanEval has prompt/canonical_solution fields
        use_streaming = False
    elif gen_config.dataset.startswith("wmt16:"):
        # Parse WMT16 configuration (e.g., "wmt16:de-en" for German-English)
        lang_pair = gen_config.dataset.split(":")[1]
        dataset = load_dataset("wmt/wmt16", lang_pair, split="train")
        dataset_format = "translation"  # WMT16 has translation field with language pairs
        use_streaming = False
        # Parse source and target languages
        src_lang, tgt_lang = lang_pair.split("-")
    else:
        dataset = load_dataset(gen_config.dataset, split="train")
        dataset_format = "qa"  # Default format with question/answer fields
        use_streaming = False

    # Set up watermarking if config is provided
    if watermark_config.strategy is not None:
        bitmap = PersistentBitmap(watermark_config.vocab_size, watermark_config.bitmap_path)
        watermark = Watermark(watermark_config, bitmap)
        # Import the custom diffusion_generate that supports watermark
        from dmark.llada.gen_dream import DreamGenerationMixin
        # We'll need to monkey-patch the model with the custom generation method
        use_custom_generation = True
    else: 
        watermark = None
        use_custom_generation = False

    # Load DREAM model
    model = AutoModel.from_pretrained(
        gen_config.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    
    # If watermarking is enabled, add the custom generation method
    if use_custom_generation:
        # Add the custom diffusion_generate method that supports watermarking
        DreamGenerationMixin.diffusion_generate.__get__(model, model.__class__)
        model.diffusion_generate = DreamGenerationMixin.diffusion_generate.__get__(model, model.__class__)
        model._sample = DreamGenerationMixin._sample.__get__(model, model.__class__)
        model._prepare_generation_config = DreamGenerationMixin._prepare_generation_config.__get__(model, model.__class__)
        model._prepare_special_tokens = DreamGenerationMixin._prepare_special_tokens.__get__(model, model.__class__)
        model._prepare_generated_length = DreamGenerationMixin._prepare_generated_length.__get__(model, model.__class__)
        model._validate_generated_length = DreamGenerationMixin._validate_generated_length.__get__(model, model.__class__)
        model._expand_inputs_for_generation = DreamGenerationMixin._expand_inputs_for_generation.__get__(model, model.__class__)
    
    tokenizer = AutoTokenizer.from_pretrained(gen_config.model, trust_remote_code=True)

    results = []
    dataset_idx = 0
    pbar = tqdm.tqdm(total=expr_config.num_samples, desc="Collecting valid samples")
    
    try:
        while len(results) < expr_config.num_samples:
            # Get next sample based on dataset format
            if dataset_format == "text":
                # Handle text-based datasets (like C4)
                try:
                    if use_streaming:
                        sample = next(dataset_iter)
                    else:
                        if dataset_idx >= len(dataset):
                            print(f"Dataset exhausted at index {dataset_idx}.")
                            break
                        sample = dataset[dataset_idx]
                        dataset_idx += 1
                except StopIteration:
                    print("Dataset exhausted.")
                    break
                
                # Get text and tokenize it
                text = sample["text"]
                text_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
                
                # Skip if text is too short (need at least 30 tokens for prompt + gen_length for ground truth)
                min_required_tokens = 30 + gen_config.gen_length
                if len(text_tokens) < min_required_tokens:
                    continue
                
                # Take first 30 tokens as prompt, next gen_length tokens as ground truth
                prompt_ids = text_tokens[:30]
                gt_ids = text_tokens[30:30 + gen_config.gen_length]
                
                # Decode prompt and ground truth
                prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                gt = tokenizer.decode(gt_ids, skip_special_tokens=True)
                
                # For text datasets, use prompt directly without chat template
                input_ids = prompt_ids.unsqueeze(0).to(device)
                attention_mask = torch.ones_like(input_ids)
                
            elif dataset_format == "code":
                # Handle code datasets (like HumanEval)
                if dataset_idx >= len(dataset):
                    print(f"Dataset exhausted at index {dataset_idx}.")
                    break
                
                sample = dataset[dataset_idx]
                dataset_idx += 1
                
                # Get prompt (function signature + docstring)
                code_prompt = sample["prompt"]
                # Store canonical solution as ground truth for reference (not for matching)
                gt = sample["canonical_solution"]
                
                # Create instruction prompt for code completion
                prompt_content = f"Complete the following Python function:\n\n{code_prompt}"
                
                # For code generation, use chat template for better instruction following
                messages = [
                    {"role": "user", "content": prompt_content},
                ]
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
                )
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                
            elif dataset_format == "translation":
                # Handle translation datasets (like WMT16)
                if dataset_idx >= len(dataset):
                    print(f"Dataset exhausted at index {dataset_idx}.")
                    break
                
                sample = dataset[dataset_idx]
                dataset_idx += 1
                
                # Get source and target texts from translation field
                translation = sample["translation"]
                source_text = translation[src_lang]
                target_text = translation[tgt_lang]
                
                # Create translation instruction prompt
                lang_names = {
                    "cs": "Czech", "de": "German", "fi": "Finnish",
                    "ro": "Romanian", "ru": "Russian", "tr": "Turkish",
                    "en": "English"
                }
                src_lang_name = lang_names.get(src_lang, src_lang)
                tgt_lang_name = lang_names.get(tgt_lang, tgt_lang)
                
                # Format as instruction for translation
                prompt_content = f"Translate the following text from {src_lang_name} to {tgt_lang_name}:\n\n{source_text}\n\nTranslation:"
                gt = target_text
                
                # For translation, use chat template for better instruction following
                messages = [
                    {"role": "user", "content": prompt_content},
                ]
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
                )
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                
            else:
                # Handle QA-based datasets (like ELI5)
                if dataset_idx >= len(dataset):
                    print(f"Dataset exhausted at index {dataset_idx}.")
                    break
                    
                question = dataset[dataset_idx]["question"]
                gt = dataset[dataset_idx]["answer"]
                messages = [
                    {"role": "user", "content": question},
                ]
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
                )
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                dataset_idx += 1

            # Generate using DREAM's diffusion_generate method
            generation_kwargs = {
                "max_new_tokens": gen_config.gen_length,
                "output_history": gen_config.output_history,
                "return_dict_in_generate": gen_config.return_dict_in_generate,
                "steps": gen_config.steps,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "alg": gen_config.alg,
                "alg_temp": gen_config.alg_temp,
                "eps": gen_config.eps,
            }
            
            if gen_config.top_k is not None:
                generation_kwargs["top_k"] = gen_config.top_k
            
            # Add watermark if using custom generation
            if use_custom_generation and watermark is not None:
                generation_kwargs["watermark"] = watermark
                
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )

            # Extract generated tokens
            if gen_config.return_dict_in_generate:
                output_ids = output.sequences[0, input_ids.shape[1]:]
            else:
                output_ids = output[0, input_ids.shape[1]:]
            
            # Trim output_ids at first occurrence of EOS token
            trimmed_length = len(output_ids)
            if tokenizer.eos_token_id is not None:
                for i, curr_token in enumerate(output_ids):
                    if curr_token == tokenizer.eos_token_id:
                        trimmed_length = i
                        break
            output_ids = output_ids[:trimmed_length]
            
            # Decode the trimmed output_ids
            if trimmed_length > 0:
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            else:
                output_text = ""
            
            # Check if output meets minimum token requirement
            num_output_tokens = output_ids.shape[0]
            if expr_config.minimum_output_token is None or num_output_tokens >= expr_config.minimum_output_token:
                results.append(
                    {
                        "data": 
                        {
                            "prompt": prompt,
                            "ground_truth": gt,
                            "output": output_text,
                            "output_ids": output_ids.tolist(),
                            "num_output_tokens": num_output_tokens,
                        },
                        "generation_metadata": gen_config.model_dump(),
                        "watermark_metadata": watermark_config.model_dump() if watermark_config.strategy is not None else None,
                    }
                )
                pbar.update(1)
            else:
                pbar.set_postfix({"skipped": f"{num_output_tokens} < {expr_config.minimum_output_token}"})
            
            dataset_idx += 1
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Collected {len(results)} samples so far.")
        pbar.close()
        return results
    
    pbar.close()
    
    if len(results) < expr_config.num_samples:
        print(f"Warning: Only collected {len(results)} valid samples out of {expr_config.num_samples} requested.")
        print(f"Dataset exhausted at index {dataset_idx}.")

    return results

if __name__ == "__main__":
    gen_config, watermark_config, expr_config = parse_args()
    results = run_generation(gen_config, watermark_config, expr_config)
    
    # Save results if any were collected
    if results:
        if expr_config.output_dir is not None:
            os.makedirs(expr_config.output_dir, exist_ok=True)
            output_filename = generate_result_filename(gen_config, watermark_config, expr_config)
            output_filename = os.path.join(expr_config.output_dir, output_filename)
            # Save results to file
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to: {output_filename}")
            print(f"Total samples collected: {len(results)}")
        else: 
            print(results)
    else:
        print("No results were collected.")