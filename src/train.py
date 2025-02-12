from tqdm import tqdm
from validate import validate
from tensorboard_logger import log_train_loss, log_validation_loss, close_writer
from config import *

# Early Stopping 관련 변수
early_stopping_patience = 10
best_loss = float('inf')
no_improvement_epochs = 0

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    global best_loss, no_improvement_epochs

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).squeeze(1).long()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 텐서보드
            log_train_loss(loss.item(), epoch, progress_bar.n, train_loader)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation 실행
        avg_val_loss, mean_iou, pixel_accuracy, best_loss, no_improvement_epochs = validate(
            model, valid_loader, criterion, num_classes, best_loss, no_improvement_epochs, class_names
        )

        # 텐서보드
        log_validation_loss(avg_val_loss, epoch)

        # 10 에폭마다 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/Segformer_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}.")

        # Early Stopping 체크
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered. Training terminated.")
            break
