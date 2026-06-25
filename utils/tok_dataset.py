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
        self.field = field  # For chunking cache, this field may be incorrect
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info
        # Removed unnecessary type checks to avoid interference

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        row_data = self.text_dataset[i]
        # ------------------------------------------------------------------
        # Core fix: prioritize pre-tokenized 'input_ids'
        # ------------------------------------------------------------------

        if 'input_ids' in row_data:
            # Path 1: data is already tokenized (from chunking cache)
            token_list = row_data['input_ids']
        else:
            # Path 2: data is raw text (legacy path or untokenized dataset)
            # Key to resolving KeyError on 'text'

            # 1. Extract raw text
            if self.field in row_data:
                text = row_data[self.field]
            else:
                # Assume row_data is a plain string
                text = row_data

                # 2. Tokenize the text
            token_list = self.tokenizer.encode(
                text, truncation=True, max_length=self.maxlen
            )
        # ------------------------------------------------------------------
        # Shared post-processing logic
        # ------------------------------------------------------------------
        # Truncate if maxlen is still needed (even after chunking), or token_list exceeds maxlen
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
