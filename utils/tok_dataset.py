import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied direcly to the model.
    """

    def __init__(self, text_dataset, tokenizer=None, maxlen=None, field="text"):
        self.text_dataset = text_dataset
        self.field = field  # 对于 Chunking 缓存，这个 field 可能是错误的
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info
        # 移除了不必要的类型检查，避免干扰

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        row_data = self.text_dataset[i]
        # ------------------------------------------------------------------
        # 核心修正逻辑：优先处理已分词的 'input_ids'
        # ------------------------------------------------------------------

        if 'input_ids' in row_data:
            # 路径 1: 数据已分词 (来自 Chunking 缓存)
            token_list = row_data['input_ids']
        else:
            # 路径 2: 数据是原始文本 (旧逻辑或未分词的数据集)
            # 解决 KeyError: 'text' 的关键点

            # 1. 提取原始文本
            if self.field in row_data:
                text = row_data[self.field]
            else:
                # 假设 row_data 是一个单独的字符串
                text = row_data

                # 2. 对文本进行分词
            token_list = self.tokenizer.encode(
                text, truncation=True, max_length=self.maxlen
            )
        # ------------------------------------------------------------------
        # 共同的后处理逻辑
        # ------------------------------------------------------------------
        # 如果 maxlen 仍需要截断（尽管 Chunking 已经处理），或者 token_list 长度超过 maxlen
        if self.maxlen is not None:
            token_list = token_list[: self.maxlen]
        position_ids = list(range(len(token_list)))
        attention_mask = [1] * len(token_list)
        return dict(
            input_ids=torch.tensor(token_list, dtype=torch.long),
            position_ids=torch.tensor(position_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )


def dict_to_(data, device):
    """
    Moves a dictionary of tensors to the specified device.
    """
    for k in data:
        data[k] = data[k].to(device)
    return data


def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)
        for k, v in items[0].items()
    }


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]
