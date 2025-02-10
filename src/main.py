import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
from train import train
from validate import validate
from config import *
from dataset import train_dataset, valid_dataset

model = smp.Segformer(encoder_name='mit_b3', encoder_weights='imagenet', classes=num_classes, activation=None).to(
    device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Early Stopping 관련 변수
best_loss = float('inf')
no_improvement_epochs = 0
best_model_state = None

# run
try:
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=1)

        avg_loss, mean_iou, pixel_accuracy, best_loss, no_improvement_epochs = validate(
            model, valid_loader, criterion, num_classes, best_loss, no_improvement_epochs, class_names
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_epochs = 0
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1} with loss {best_loss:.4f}.")

        # ✅ 10 에폭마다 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/Segformer_epoch_{epoch + 1}.pth")
            print(f"Model saved at epoch {epoch + 1}.")

        # ✅ Early stopping 체크
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping trig  gered. Training terminated.")
            break

except KeyboardInterrupt:
    # ✅ 학습 중단 시점의 가중치 저장
    print("\nTraining interrupted. Saving current model state...")
    torch.save(model.state_dict(), f"{save_dir}/current_Segformer_epoch_{epoch+1}.pth")

# ✅ 최종 모델 저장
torch.save(model.state_dict(), f"{save_dir}/final_Segformer.pth")
print("Final model saved.")

# ✅ 가장 성능 좋은 모델 저장
if best_model_state is not None:
    torch.save(best_model_state, f"{save_dir}/best_Segformer.pth")
    print("Final best model saved.")
