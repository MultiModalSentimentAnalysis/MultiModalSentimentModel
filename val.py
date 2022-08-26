import evaluate, torch

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
precision_macro = evaluate.load("precision")
precision_micro = evaluate.load("precision")
precision_weighted = evaluate.load("precision")
f1 = evaluate.load("f1")
f1_macro = evaluate.load("f1")
f1_micro = evaluate.load("f1")
f1_weighted = evaluate.load("f1")
recall = evaluate.load("recall")
recall_macro = evaluate.load("recall")
recall_micro = evaluate.load("recall")
recall_weighted = evaluate.load("recall")


def validate(model, dataloader, loss_fn, language="eng"):
    running_loss = 0.0

    for data_pair_index, batch in enumerate(dataloader):
        # print("--------------", data_pair_index, "-------------")
        errors = (batch["pose_embedding"] == -123).all(dim=1)
        assert torch.all(~errors).item()

        text_embedding = batch[f"{language}_text_embedding"]
        face_embedding = batch["face_embedding"]
        pose_embedding = batch["pose_embedding"]
        scene_embedding = batch["scene_embedding"]
        labels = batch["sentiment"]
        inputs = torch.cat(
            (face_embedding, text_embedding, pose_embedding, scene_embedding), 1
        )
        logits = model(inputs)
        probs = logits.argmax(dim=1)

        accuracy.add_batch(predictions=probs, references=labels)
        precision.add_batch(predictions=probs, references=labels)
        precision_macro.add_batch(predictions=probs, references=labels)
        precision_micro.add_batch(predictions=probs, references=labels)
        precision_weighted.add_batch(predictions=probs, references=labels)
        f1.add_batch(predictions=probs, references=labels)
        f1_macro.add_batch(predictions=probs, references=labels)
        f1_micro.add_batch(predictions=probs, references=labels)
        f1_weighted.add_batch(predictions=probs, references=labels)
        recall.add_batch(predictions=probs, references=labels)
        recall_macro.add_batch(predictions=probs, references=labels)
        recall_micro.add_batch(predictions=probs, references=labels)
        recall_weighted.add_batch(predictions=probs, references=labels)

        loss = loss_fn(logits, labels)
        running_loss += loss.item()

    print(accuracy.compute())
    print(precision.compute(average=None))
    print(precision_macro.compute(average="macro"))
    print(precision_micro.compute(average="micro"))
    print(precision_weighted.compute(average="weighted"))
    print(f1.compute(average=None))
    print(f1_macro.compute(average="macro"))
    print(f1_micro.compute(average="micro"))
    print(f1_weighted.compute(average="weighted"))
    print(recall.compute(average=None))
    print(recall_macro.compute(average="macro"))
    print(recall_micro.compute(average="micro"))
    print(recall_weighted.compute(average="weighted"))
