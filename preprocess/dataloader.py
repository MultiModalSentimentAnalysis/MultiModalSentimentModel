from torch.utils.data import DataLoader
from datasets import MSCTDDataSet



class MSCTDDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
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
    batch_size = 10
    device = "cuda"
    ds = MSCTDDataSet(base_path="data/", dataset_type = "val", load=True)
    ds.save_features()
    ds.load_data()
    dl = DataLoader(ds, batch_size=batch_size)
    dl = MSCTDDataLoader(dl, device)
    for x in dl:
        print(x)
        print(x['face_embedding'].shape)
        print(x['text_embedding'].shape)
        print(x['real_index'])
