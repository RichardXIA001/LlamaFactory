from openai import OpenAI
import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from peft import PeftModel


def get_vl_models():
    """Fetch available VL models from DashScope API."""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        models = client.models.list()
        vl_models = [model.id for model in models.data if 'vl' in model.id.lower()]
        
        print("Available VL models:")
        for model in vl_models:
            print(f"  - {model}")
        
        return vl_models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None


def load_local_qwen3vl_model(
    model_name="qwen3-vl-4b-instruct",
    lora_path=None,
    is_lora=False,
    device="auto",
    dtype="auto",
    model_scope=False,
    enable_hf_mirror=False,
    hf_token=None,
    merge_lora=True
):
    """
    Load Qwen3-VL model with optional LoRA adapter.
    
    Args:
        model_name: Base model name or path (e.g., "Qwen/Qwen3-VL-4B-Instruct")
        lora_path: Path to LoRA adapter (e.g., "saves/qwen3-vl-4b/lora/sft-sku_110k/checkpoint-500")
        is_lora: Whether to load LoRA adapter
        device: Device to load model on ("auto", "cuda", "cpu")
        dtype: Data type ("auto", "float16", "bfloat16", "float32")
        model_scope: Whether to use ModelScope for base model download
        enable_hf_mirror: Enable HuggingFace mirror
        hf_token: HuggingFace token
        merge_lora: Whether to merge LoRA weights into base model (recommended for inference)
    
    Returns:
        (model, processor, device_str)
    """
    
    print("=" * 70)
    print("QWEN3-VL MODEL LOADER")
    print("=" * 70)
    
    # Setup environment
    if enable_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resolve dtype
    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, dtype)
    
    print(f"Device: {device} | Dtype: {dtype}")
    
    # If LoRA is enabled, extract base model from adapter_config.json
    if is_lora:
        if not lora_path:
            raise ValueError("lora_path must be provided when is_lora=True")
        
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise ValueError(f"LoRA path does not exist: {lora_path}")
        
        # Read adapter_config.json to get base model
        adapter_config_path = lora_path / "adapter_config.json"
        if not adapter_config_path.exists():
            raise ValueError(f"adapter_config.json not found in {lora_path}")
        
        print(f"→ Reading adapter config from: {adapter_config_path}")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Extract base model path
        base_model_from_config = adapter_config.get("base_model_name_or_path")
        if not base_model_from_config:
            raise ValueError(
                f"'base_model_name_or_path' not found in adapter_config.json. "
                f"Please specify the base model manually."
            )
        model_name = base_model_from_config
        print(f"→ Base model from config: {base_model_from_config}")
        
    # Load base model
    print(f"→ Loading base model: {model_name}")
    
    if model_scope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_name, cache_dir=None)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Determine model class
    if type(config) in AutoModelForImageTextToText._model_mapping.keys():
        load_class = AutoModelForImageTextToText
    elif type(config) in AutoModelForVision2Seq._model_mapping.keys():
        load_class = AutoModelForVision2Seq
    else:
        load_class = AutoModelForCausalLM
    
    # Load base model
    model = load_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter if specified
    if is_lora:
        print(f"→ Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))
        
        if merge_lora:
            print(f"→ Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print(f"✓ LoRA weights merged successfully")
        else:
            print(f"✓ LoRA adapter loaded (not merged)")
    
    model.eval()
    
    print("=" * 70)
    print(f"✓ Model loaded successfully!")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Device: {device}")
    print("=" * 70)
    
    return model, processor, device


# Usage examples
if __name__ == "__main__":
    
    # # Example 1: Load base model only
    # print("\n🔹 Example 1: Load Base Model")
    # model, processor, device = load_local_qwen3vl_model(
    #     model_name="Qwen/Qwen3-VL-4B-Instruct",
    #     is_lora=False
    # )
    # print(f"Model loaded on {device}!\n")
    # del model, processor
    
    # Example 2: Load base model from ModelScope
    
    # Example 3: Load LoRA model (base model from adapter_config.json)
    print("\n🔹 Example 3: Load LoRA Model (auto-detect base model)")
    model, processor, device = load_local_qwen3vl_model(
        lora_path="/root/Codes/LlamaFactory/saves/qwen3-vl-4b/lora/sft-sku_110k/checkpoint-500",
        is_lora=True,
        merge_lora=True
    )
    print(f"LoRA model loaded on {device}!\n")
    del model, processor
    
    # Example 4: Load LoRA model with explicit base model
    print("\n🔹 Example 4: Load LoRA Model (explicit base model)")
    model, processor, device = load_local_qwen3vl_model(
        model_name="Qwen/Qwen3-VL-4B-Instruct",  # Explicit base model
        lora_path="/root/Codes/LlamaFactory/saves/qwen3-vl-4b/lora/sft-sku_110k/checkpoint-771",
        is_lora=True,
        merge_lora=True
    )
    print(f"LoRA model loaded on {device}!\n")
    del model, processor
    
    # Example 5: Load LoRA without merging (memory efficient)
    print("\n🔹 Example 5: Load LoRA Model (without merging)")
    model, processor, device = load_local_qwen3vl_model(
        lora_path="/root/Codes/LlamaFactory/saves/qwen3-vl-4b/lora/sft-sku_110k/checkpoint-500",
        is_lora=True,
        merge_lora=False
    )
    print(f"LoRA model loaded on {device}!\n")
    del model, processor
    
    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)