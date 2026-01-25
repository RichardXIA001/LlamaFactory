from openai import OpenAI
import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
)


def get_vl_models():
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
    device="auto",
    dtype="auto",
    model_scope=False,
    enable_hf_mirror=False,
    hf_token=None
):
    """
    Load local Qwen3-VL model + processor once.

    Returns:
        model, processor, device_str
    """


    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve dtype
    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            # bfloat16 is generally stable on Ampere+; switch to float16 if you prefer
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    else:
        # allow strings like "float16"/"bfloat16"/"float32"
        dtype = getattr(torch, dtype)

    if enable_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

    if model_scope:
        from modelscope import snapshot_download  # type: ignore
        model_path = snapshot_download(model_name, cache_dir=None)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if type(config) in AutoModelForImageTextToText._model_mapping.keys():
        load_class = AutoModelForImageTextToText
    elif type(config) in AutoModelForVision2Seq._model_mapping.keys():
        load_class = AutoModelForVision2Seq
    else:
        load_class = AutoModelForCausalLM

    model = load_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,          # keep consistent with your script
        trust_remote_code=True,
    )
    model.eval()

    return model, processor, device


# Usage
if __name__ == "__main__":
    vl_models = get_vl_models()
    print("Available VL models:")
    for model in vl_models:
        print(f"  - {model}")
    model, processor, device = load_local_qwen3vl_model(model_name="Qwen/Qwen3-VL-8B-Instruct", model_scope=True)
    print(f"Model loaded successfully on {device}!")
    print(f"Processor loaded successfully!")
    print(f"Model type: {model.config.model_type}")
    print(f"Model config: {model.config}")