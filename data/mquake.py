import json
from pathlib import Path
from torch.utils.data import Dataset
from utils.globals import *


# 辅助函数：获取带模板的 Prompt
def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_im_end|>\n<|im_start|>assistant\n"""


def get_list_llama_without_answer(que_list, answer=False):
    return [get_llama_without_answer(q) for q in que_list]


def get_list_qwen_without_answer(que_list, answer=False):
    return [get_qwen_without_answer(q) for q in que_list]


class MQUAKEDataset(Dataset):
    def __init__(self, data_dir: str, model_name: str, size: int = None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_path = data_dir / "AKEW" / "MQuAKE-CF.json"

        with open(mquake_path, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            req_rewrite = record["requested_rewrite"][0]

            item = {
                "case_id": i,
                "id": i,
                "requested_rewrite": req_rewrite,
                "questions": record["questions"],
                "new_answer": record["new_answer"],
                "new_answer_alias": record.get("new_answer_alias", []),
                "new_single_hops": record.get("new_single_hops", [])
            }

            # 模板处理保持不变
            if 'Llama-3' in model_name or 'Llama3' in model_name:
                item["question"] = get_llama_without_answer(req_rewrite["question"])
                item["answer"] = req_rewrite["fact_new_uns"]
            elif 'Qwen' in model_name:
                item["question"] = get_qwen_without_answer(req_rewrite["question"])
                item["answer"] = req_rewrite["fact_new_uns"]
            else:
                item["question"] = req_rewrite["question"]
                item["answer"] = req_rewrite["fact_new_uns"]

            data.append(item)

        if size is not None:
            self._data = data[:size]
        else:
            self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)