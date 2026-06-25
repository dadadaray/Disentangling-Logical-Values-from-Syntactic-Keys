import json
from pathlib import Path
from torch.utils.data import Dataset


def get_llama_prompt(q):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def get_qwen_without_answer(q):
    return f"""<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"""


class GSM8KDataset(Dataset):
    def __init__(self, data_dir: str, model_name: str, size: int = None, *args, **kwargs):
        data_dir = Path(data_dir)
        path = data_dir /"AKEW"/"gsm8k.json"

        with open(path, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            item = {
                "case_id": i,
                "question": record["question"],  # Raw question text
                "answer": record["answer"],  # Numeric answer string (e.g., "42")
                "prompt": get_llama_prompt(record["question"]) if "Llama-3" in model_name else record["question"]
            }
            data.append(item)

        if size is not None:
            self._data = data[:size]
        else:
            self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)