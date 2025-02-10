import torch
import segmentation_models_pytorch as smp

def load_UNet_model(weight_path, num_classes, device):
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,
        classes=num_classes,
        activation=None
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_segformer_model(weight_path, num_classes, device):
    model = smp.Segformer(
        encoder_name="mit_b3",
        encoder_weights=None,
        classes=num_classes,
        activation=None
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
