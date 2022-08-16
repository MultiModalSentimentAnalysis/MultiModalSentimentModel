from torch.utils.data import DataLoader
from datasets import MSCTDDataSet


class MSCTDDataLoader:
    def __init__(self, dl, device, tokenizer=None, text_len=512):
        self.dl = dl
        self.device = device
        self.tokenizer = tokenizer
        self.text_len = text_len

    def __iter__(self):
        for b in self.dl:
            if self.tokenizer:
                b["text"] = self.tokenizer(
                    b["text"],
                    padding="max_length",
                    max_length=self.text_len,  # including [CLS] end [SEP]
                    truncation=True,
                    return_tensors="pt",
                    # return_offsets_mapping=True,
                )
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, str):
        return data
    return data.to(device)


if __name__ == "__main__":
    batch_size = 2
    num_workers = 1

    from transformers import AutoTokenizer

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # tokenizer = None
    dataset = MSCTDDataSet("data/", "val")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader = MSCTDDataLoader(dataloader, device, tokenizer)
    # train_loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True) # WHY THIS TAKES DAYS?
    for x in dataloader:
        print(x["images"].shape)
        if tokenizer:
            print(x["text"]["input_ids"].shape)
        else:
            print(x["text"])
        print(x["sentiment"].shape)
        break
