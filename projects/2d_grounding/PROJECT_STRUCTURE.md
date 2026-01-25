# 2D Grounding Project Structure Guide

This document outlines the recommended folder structure for the `2d_grounding` module, following LlamaFactory conventions and LLM training/inference best practices.

## Recommended Directory Structure

```
2d_grounding/
├── __init__.py                 # Module initialization
├── README.md                    # Module documentation
│
├── model/                       # Model-related code
│   ├── __init__.py
│   ├── loader.py               # Model loading utilities (integrate with llamafactory.model.loader)
│   ├── inference.py            # Inference engine for 2D grounding
│   ├── patcher.py              # Model-specific patches for 2D grounding
│   └── adapter.py              # Adapter for 2D grounding models (if needed)
│
├── train/                       # Training code
│   ├── __init__.py
│   ├── trainer.py              # Main trainer for 2D grounding fine-tuning
│   ├── workflow.py             # Training workflow orchestration
│   ├── metric.py               # Training metrics (IoU, mAP, etc.)
│   └── collator.py             # Data collator for 2D grounding batches
│
├── eval/                        # Evaluation code
│   ├── __init__.py
│   ├── evaluator.py            # Main evaluator (IoU, mAP, precision/recall)
│   ├── metrics.py              # Metric computation functions
│   └── dataset_loader.py       # Load evaluation datasets
│
├── data/                        # Data processing
│   ├── __init__.py
│   ├── dataset.py              # Dataset class for 2D grounding
│   ├── converter.py            # Convert datasets to LlamaFactory format
│   ├── collator.py             # Data collation for training
│   └── template.py             # Prompt templates for 2D grounding
│
├── utils/                       # Utility functions (already exists)
│   ├── __init__.py
│   ├── box_utils.py            # Bounding box utilities
│   ├── draw_utils.py           # Visualization utilities
│   ├── image_utils.py          # Image loading/processing
│   └── parse_utils.py          # Response parsing utilities
│
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── model_config.py         # Model configuration defaults
│   └── prompts.py              # Prompt templates and configurations
│
├── scripts/                     # Executable scripts
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Evaluation entry point
│   ├── infer.py                # Inference entry point (move from root)
│   └── convert_dataset.py      # Dataset conversion script
│
├── examples/                    # Example configurations
│   ├── train_lora/
│   │   └── qwen3vl_2d_grounding_lora.yaml
│   ├── train_full/
│   │   └── qwen3vl_2d_grounding_full.yaml
│   └── inference/
│       └── qwen3vl_2d_grounding_inference.yaml
│
└── tests/                       # Unit and integration tests
    ├── __init__.py
    ├── test_model/
    │   ├── test_loader.py
    │   └── test_inference.py
    ├── test_train/
    │   └── test_trainer.py
    ├── test_eval/
    │   └── test_evaluator.py
    ├── test_data/
    │   └── test_dataset.py
    └── test_utils/
        ├── test_box_utils.py
        └── test_parse_utils.py
```

## Detailed File Descriptions

### 1. `model/` - Model Management

**`loader.py`**: 
- Integrate with `llamafactory.model.loader` 
- Add 2D grounding-specific model loading logic
- Handle Qwen3VL and other vision-language models
- Support LoRA/QLoRA loading for fine-tuned models

**`inference.py`**:
- Batch inference engine
- Model generation with proper prompts
- Response post-processing
- Integration with existing inference pipeline

**`patcher.py`**:
- Model-specific patches for 2D grounding tasks
- Custom attention mechanisms if needed
- Vision encoder modifications

### 2. `train/` - Training Pipeline

**`trainer.py`**:
- Extend or use `llamafactory.train.tuner`
- Custom loss functions for bounding box regression
- Training loop with 2D grounding-specific logic
- Integration with existing training infrastructure

**`workflow.py`**:
- Training workflow orchestration
- Data loading → Training → Validation → Checkpointing
- Follow pattern from `llamafactory.train.sft.workflow`

**`metric.py`**:
- Training-time metrics (IoU, loss tracking)
- Logging to TensorBoard/W&B
- Custom metric callbacks

**`collator.py`**:
- Batch collation for images + text + bounding boxes
- Padding and batching strategies
- Multi-image handling

### 3. `eval/` - Evaluation

**`evaluator.py`**:
- Main evaluation class (similar to `llamafactory.eval.evaluator`)
- Run inference on test set
- Compute metrics (mAP, IoU, precision/recall)
- Generate evaluation reports

**`metrics.py`**:
- IoU calculation
- mAP (mean Average Precision) computation
- Precision/Recall/F1 metrics
- COCO-style evaluation metrics

**`dataset_loader.py`**:
- Load standard evaluation datasets (COCO, SKU110K, etc.)
- Format conversion for evaluation

### 4. `data/` - Data Processing

**`dataset.py`**:
- PyTorch Dataset class for 2D grounding
- Image loading and preprocessing
- Label formatting (bounding boxes + text)
- Support for various dataset formats

**`converter.py`**:
- Convert datasets to LlamaFactory format
- Support SKU110K, COCO, custom formats
- Generate dataset_info.json entries

**`collator.py`**:
- Data collation for training batches
- Handle variable-length sequences
- Image and text batching

**`template.py`**:
- Prompt templates for 2D grounding
- System prompts, user prompts
- Format conversion (text → bbox)

### 5. `utils/` - Utilities (Already Exists)

Keep existing utilities:
- `box_utils.py`: Bounding box operations
- `draw_utils.py`: Visualization
- `image_utils.py`: Image loading
- `parse_utils.py`: Response parsing

### 6. `config/` - Configuration

**`model_config.py`**:
- Default model configurations
- Model-specific hyperparameters
- Architecture settings

**`prompts.py`**:
- Prompt templates
- System messages
- Task-specific prompts

### 7. `scripts/` - Executable Scripts

**`train.py`**:
- Training entry point
- Parse arguments
- Initialize trainer
- Follow `src/train.py` pattern

**`eval.py`**:
- Evaluation entry point
- Load model and dataset
- Run evaluation
- Output results

**`infer.py`**:
- Move `detect_objects_qwen3vl_inference.py` here
- Rename to `infer.py` for consistency
- Clean up and modularize

**`convert_dataset.py`**:
- Dataset conversion utility
- CLI for converting datasets

### 8. `examples/` - Configuration Examples

**YAML configs** following LlamaFactory pattern:
- Model configuration
- Training hyperparameters
- Dataset settings
- Output directories

Example structure:
```yaml
### model
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
image_max_pixels: 262144
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8

### dataset
dataset: sku_110k_2d_grounding
template: qwen3_vl_2d_grounding
cutoff_len: 2048

### output
output_dir: saves/qwen3-vl-4b/lora/2d-grounding-sku110k
```

### 9. `tests/` - Testing

Follow LlamaFactory testing patterns:
- Unit tests for each module
- Integration tests for training/eval pipelines
- Use pytest
- Mock external dependencies

## Integration with LlamaFactory

### 1. Model Loading
- Integrate with `llamafactory.model.loader.load_model()`
- Add 2D grounding-specific model classes if needed
- Support existing adapter system (LoRA, QLoRA)

### 2. Training
- Extend `llamafactory.train.tuner` or create custom trainer
- Use existing hyperparameter system (`llamafactory.hparams`)
- Integrate with training callbacks

### 3. Data Processing
- Use `llamafactory.data` infrastructure
- Add custom dataset to `data/dataset_info.json`
- Support existing data formatters

### 4. Evaluation
- Follow `llamafactory.eval.evaluator` pattern
- Integrate with existing evaluation framework
- Support standard metrics

## Migration Plan

1. **Phase 1: Organize existing code**
   - Move `detect_objects_qwen3vl_inference.py` → `scripts/infer.py`
   - Organize utilities in `utils/`
   - Create empty structure for new modules

2. **Phase 2: Implement core modules**
   - Implement `model/loader.py` and `model/inference.py`
   - Create `data/dataset.py` and `data/converter.py`
   - Set up `train/trainer.py` skeleton

3. **Phase 3: Training pipeline**
   - Implement training workflow
   - Add data collation
   - Create example configs

4. **Phase 4: Evaluation**
   - Implement evaluator
   - Add metrics computation
   - Create evaluation scripts

5. **Phase 5: Testing and documentation**
   - Write unit tests
   - Add integration tests
   - Update documentation

## Best Practices

1. **Follow LlamaFactory patterns**: Use existing infrastructure where possible
2. **Modular design**: Separate concerns (model, train, eval, data)
3. **Configuration-driven**: Use YAML configs like main LlamaFactory
4. **Type hints**: Use type annotations throughout
5. **Documentation**: Add docstrings following Google style
6. **Testing**: Write tests for critical functionality
7. **License headers**: Include Apache 2.0 license headers
8. **Code style**: Follow LlamaFactory style guide (ruff, 119 char line length)

## Example Integration Points

### Using LlamaFactory Model Loader
```python
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import get_train_args

# In your model/loader.py
def load_2d_grounding_model(model_args, finetuning_args):
    tokenizer_dict = load_tokenizer(model_args)
    model = load_model(tokenizer_dict["tokenizer"], model_args, finetuning_args)
    return model, tokenizer_dict
```

### Using LlamaFactory Training Infrastructure
```python
from llamafactory.train import get_train_args
from llamafactory.train.tuner import run_exp

# In your train/trainer.py
def train_2d_grounding(args):
    model_args, data_args, training_args, finetuning_args = get_train_args(args)
    # Custom 2D grounding training logic
    run_exp(...)
```

## Next Steps

1. Review this structure
2. Create the directory structure
3. Move existing files to appropriate locations
4. Implement core modules incrementally
5. Add tests as you develop
6. Create example configurations
7. Document the module
