import torch
from sklearn import metrics


def intersection_over_union(boxes_preds, boxes_true):  # takes lists

    box = torch.tensor((boxes_preds, boxes_true))
    box = box.view(-1, 4)
    boxes = box_to_corner(box)

    counted_boxes = torch.tensor(torch.numel(boxes) / 4 / 2).int()

    boxes_preds = boxes.data[:counted_boxes]
    boxes_true = boxes.data[counted_boxes:]

    x1 = torch.max(boxes_preds[:, 0:1], boxes_true[:, 0:1])
    y1 = torch.max(boxes_preds[:, 1:2], boxes_true[:, 1:2])
    x2 = torch.min(boxes_preds[:, 2:3], boxes_true[:, 2:3])
    y2 = torch.min(boxes_preds[:, 3:4], boxes_true[:, 3:4])

    # clamp(0) - if they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((boxes_preds[:, 2:3] - boxes_preds[:, 0:1]) * (boxes_preds[:, 1:2] - boxes_preds[:, 3:4]))
    box2_area = abs((boxes_true[:, 2:3] - boxes_true[:, 0:1]) * (boxes_true[:, 1:2] - boxes_true[:, 3:4]))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def box_to_corner(boxes):  # takes tensors
    # Convert to (upper-left, lower-right)
    x0, y0, h, w = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x0
    y1 = y0
    x2 = x0 + w
    y2 = y0 + h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def sublist(main_list):
    result = []

    for sublist in main_list:
        for item in sublist:
            result.append(item)

    return result


def metrics(TP, TN, FP, FN):
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0

    return recall, precision, f1


actual_list = []
predictions_list = []


def inference(model, dataloader, device):
    correct_prediction = 0
    total_prediction = 0
    predictions = []
    actual = []

    # modelis rado medi ir jis ten buvo
    # kai IoU > 0.5 && klase atitinka
    TP = 0
    # modelis rado medi ir ten jo nebuvo:
    # kai IoU < 0.5 ir klase atitinka
    TN = 0
    # modelis nerado medzio ir modelis suklydo:
    # kai IoU < 0.5 ir klase neatitinka
    FP = 0
    # modelis nerado medzio ir nesuklydo:
    # kai IoU > 0.5 ir klase neatitinka
    FN = 0

    with torch.no_grad():
        for data in dataloader:

            input_images, labels, coords = data[0].to(device), data[1].to(device), data[2].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = input_images.mean(), input_images.std()
            input_images = (input_images - inputs_m) / inputs_s

            output_class, output_bb = model(input_images)

            _, prediction_class = torch.max(output_class, 1)
            print(f"predicted classes: {prediction_class}")
            print(f"true classes: {labels}")
            print(f"predicted coordinates: {output_bb}")
            print(f"true coordinates: {coords}")

            IoU_tensor = intersection_over_union(output_bb.tolist(), coords.tolist())

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

            total_prediction += prediction_class.shape[0]

    recall, precision, f1 = metrics(TP, TN, FP, FN)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    acc = TP / total_prediction

    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1: {f1}')
