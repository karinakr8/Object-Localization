import torch
import inference


def train_single_epoch(model, dataloader, loss_bb, loss_class, optimizer, device, scheduler):
    running_loss = 0.0
    total_prediction = 0

    # IoU > 0.5 && class
    TP = 0
    # IoU < 0.5 && class
    TN = 0
    # IoU < 0.5 && !class
    FP = 0
    # IoU > 0.5 && !class
    FN = 0

    for x, data in enumerate(dataloader):

        input_images, labels, coords = data[0].to(device), data[1].to(device), data[2].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = input_images.mean(), input_images.std()
        input_images = (input_images - inputs_m) / inputs_s

        output_class, output_bb = model(input_images)

        optimizer.zero_grad()

        # forward + backward + optimize
        loss_c = loss_class(output_class, labels)
        loss_b = loss_bb(output_bb, coords)
        loss = loss_c + loss_b

        # loss = loss_fn(output_bb, input_images)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # -----------------------------------------------------------------
        _, prediction_class = torch.max(output_class, 1)
        # print(f"predicted classes: {prediction_class}")
        # print(f"true classes: {labels}")

        # print(f"predicted coordinates: {output_bb}")
        # print(f"true coordinates: {coords}")
        # -----------------------------------------------------------------
        total_prediction += prediction_class.shape[0]

        IoU_tensor = inference.intersection_over_union(output_bb.tolist(), coords.tolist())

        print(f"Correct coords: {(IoU_tensor > 0.5).sum().item()}/16")
        print(f"Correct classes: {(prediction_class == labels).sum().item()}/16")

        print(f"IoU: {IoU_tensor}")

        indexes = []

        for i, _ in enumerate(prediction_class):
            if prediction_class[i] == labels[i]:
                indexes.append(i)

        for i, _ in enumerate(IoU_tensor):
            if i in indexes:  # class meets
                if IoU_tensor[i] > 0.5:  # coords meets
                    TP += 1
                else:  # coords not meets
                    TN += 1
            else:  # class not meets
                if IoU_tensor[i] > 0.5:  # coords meets
                    FN += 1
                else:  # coords not meets
                    FP += 1

    recall, precision, f1 = inference.metrics(TP, TN, FP, FN)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    acc = TP / total_prediction
    print(f'Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Total items: {total_prediction}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1: {f1}')


def train(model, data_loader, loss_bb, loss_class, optimiser, device, epochs, scheduler):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_bb, loss_class, optimiser, device, scheduler)
        print("---------------------------")
    print("Finished training")
