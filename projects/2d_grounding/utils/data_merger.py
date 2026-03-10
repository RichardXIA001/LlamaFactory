import json
from pathlib import Path
from typing import Sequence


def merge_json_files(base_dir: Path, input_template: str, indices: Sequence[int], output_path: Path):
    merged = []

    for i in indices:
        path = base_dir / input_template.format(i)
        print(f"Loading {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            merged.extend(data)
        else:
            raise TypeError(f"Expected list in {path}, got {type(data)}")

    print(f"Saving merged file to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    base_dir = Path("/root/Codes/LlamaFactory/projects/2d_grounding/results/rft/training_images_run_260309")
    input_template = "sft_rft_gpu{}.json"
    indices = range(8)
    output_path = base_dir / "sft_rft_merged_0-7.json"

    merge_json_files(base_dir, input_template, indices, output_path)
