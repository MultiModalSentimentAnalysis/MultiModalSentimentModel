import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from preprocess.datasets import MSCTDDataSet
from settings import SAVE_DIR


SAVE_SPLIT = "test"
SAVE_BATCH = 8
dataset = MSCTDDataSet(split=SAVE_SPLIT, load=False)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=SAVE_BATCH)


def save_features(dataloader, split):
    """
    Save features using a dataset with load = False. Later on you can use dataset with load=True for a
    fast dataset for later trainings.
    """
    stop_batch = None

    for batch_index, batch in enumerate(tqdm(dataloader)):
        errors = (batch["pose_embedding"] == -123).all(dim=1)

        torch.save(
            batch["face_embedding"][~errors],
            SAVE_DIR / f"face_embeddings_{split}_{batch_index}.pt",
        )
        torch.save(
            batch["pose_embedding"][~errors],
            SAVE_DIR / f"pose_embeddings_{split}_{batch_index}.pt",
        )
        torch.save(
            batch["eng_text_embedding"][~errors],
            SAVE_DIR / f"eng_text_embeddings_{split}_{batch_index}.pt",
        )
        torch.save(
            batch["ger_text_embedding"][~errors],
            SAVE_DIR / f"ger_text_embeddings_{split}_{batch_index}.pt",
        )
        torch.save(
            batch["index"][~errors],
            SAVE_DIR / f"indexes_{split}_{batch_index}.pt",
        )
        torch.save(
            batch["sentiment"][~errors],
            SAVE_DIR / f"sentiments_{split}_{batch_index}.pt",
        )
        torch.save(
            batch["scene_embedding"][~errors],
            SAVE_DIR / f"scene_embeddings_{split}_{batch_index}.pt",
        )

        assert (
            batch["pose_embedding"].shape[0] == batch["eng_text_embedding"].shape[0]
        ), "text and pose list are not the same size in saving"
        assert (
            batch["face_embedding"].shape[0] == batch["pose_embedding"].shape[0]
        ), "face and pose list are not the same size in saving"
        assert (
            batch["eng_text_embedding"].shape[0] == batch["index"].shape[0]
        ), "text and index list are not the same size in saving"
        assert (
            batch["index"].shape[0] == batch["sentiment"].shape[0]
        ), "index and sentiment list are not the same size in saving"
        assert (
            batch["ger_text_embedding"].shape[0] == batch["index"].shape[0]
        ), "index and sentiment list are not the same size in saving"
        if stop_batch and batch_index == stop_batch:
            break

    print("----------------------")
    len_batch = len(dataloader)
    if stop_batch:
        len_batch = stop_batch
    print(len(dataloader))

    def concat_batches(name="face_embedding"):
        batches = []
        for i in range(len_batch):
            batches.append(torch.load(SAVE_DIR / f"{name}_{split}_{i}.pt"))
        batches = torch.cat(batches, dim=0)
        print(batches.shape)
        torch.save(batches, SAVE_DIR / f"{name}_{split}.pt")
        del batches

    for name in [
        "face_embeddings",
        "pose_embeddings",
        "eng_text_embeddings",
        "ger_text_embeddings",
        "text_embeddings",
        "scene_embeddings",
        "sentiments",
        "indexes",
    ]:
        concat_batches(name)


save_features(dataloader, SAVE_SPLIT)
