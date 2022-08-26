import torch, tqdm
from torch import nn, optim
from MultiModalEmotionRecognition.models import SimpleDenseNetwork
from MultiModalEmotionRecognition.preprocess.datasets import get_dataset_and_dataloder
from MultiModalEmotionRecognition.settings import DEVICE, FULL_EMBEDDING_SIZE, SAVE_DIR
from MultiModalEmotionRecognition.val import validate

LEARNING_RATE = 0.00001
BATCH_SIZE = 32
EPOCHS = 30
MOMENTUM = 0.001
num_workers = 1


def train_epoch(epoch_index, model, dataloader, loss_fn, optimizer, language="eng"):
    running_loss = 0.0

    for batch_index, batch in enumerate(tqdm(dataloader)):
        errors = (batch["pose_embedding"] == -123).all(dim=1)
        assert torch.all(~errors).item()
        text_embedding = batch[f"{language}_text_embedding"]
        face_embedding = batch["face_embedding"]
        pose_embedding = batch["pose_embedding"]
        scene_embedding = batch["scene_embedding"]
        labels = batch["sentiment"]
        optimizer.zero_grad()

        inputs = torch.cat(
            (face_embedding, text_embedding, pose_embedding, scene_embedding), 1
        )
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Epoch loss: ", running_loss)


def train_model(model, epochs, train_dataloader, val_dataloader, language="eng"):
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs):
        print("--------------epoch: ", epoch, "-------------")
        model.train()
        train_epoch(epoch, model, train_dataloader, loss_fn, optimizer, language)
        model.eval()
        validate(model, val_dataloader, loss_fn)


train_dataset, train_dataloader = get_dataset_and_dataloder("train", BATCH_SIZE)
val_dataset, val_dataloader = get_dataset_and_dataloder("val", BATCH_SIZE)
test_dataset, test_dataloader = get_dataset_and_dataloder("test", BATCH_SIZE)

model = SimpleDenseNetwork(n_classes=3, embedding_dimension=FULL_EMBEDDING_SIZE).to(
    device=DEVICE
)

train_model(model, EPOCHS, train_dataloader, val_dataloader)

validate(model, test_dataloader, nn.CrossEntropyLoss())

torch.save(model, SAVE_DIR / "eng_model.pt")
