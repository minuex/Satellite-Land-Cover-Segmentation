import torch
from compute_metrics import compute_metrics
from config import device

def validate(model, valid_loader, criterion, num_classes, best_loss, no_improvement_epochs, class_names):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_pixel_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device).squeeze(1).long()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Compute metrics
            mean_iou, pixel_accuracy = compute_metrics(outputs, masks, num_classes, class_names)
            total_iou += mean_iou
            total_pixel_accuracy += pixel_accuracy
            num_batches += 1

    avg_loss = running_loss / len(valid_loader)
    mean_iou = total_iou / num_batches
    pixel_accuracy = total_pixel_accuracy / num_batches

    print(f"Validation Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}")

    return avg_loss, mean_iou, pixel_accuracy, best_loss, no_improvement_epochs
