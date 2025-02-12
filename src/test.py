import rasterio
import torchvision.transforms as transforms
from config import *
from model_loader import load_UNet_model, load_segformer_model
from preprocessing import *
from postprocessing import (
    create_edge_map,
    adaptive_road_filling,
    refine_road_with_edges,
    directional_road_filter,
    contour_simplification,
    uncertainty_based_road_refine
)
######################################################################################

output_folder = "predictions"
os.makedirs(output_folder, exist_ok=True)

def preprocess_image(image, image_transform):
    image = Image.fromarray(image).convert("RGB")
    image = image_transform(image)
    return image.unsqueeze(0)

def preprocess_mask(mask, class_mapping):
    mask_array = np.array(mask)
    mapped_mask = np.zeros_like(mask_array, dtype=np.uint8)
    for old_class, new_class in class_mapping.items():
        mapped_mask[mask_array == old_class] = new_class
    return torch.tensor(mapped_mask, dtype=torch.long)

def save_prediction(mask, output_path):
    """ 단일 밴드 흑백 마스크를 GeoTIFF로 저장 """
    height, width = mask.shape
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,  # 단일 채널
        dtype=rasterio.uint8,  # 데이터 타입: 8비트 정수 (0-255)
        crs="EPSG:32652",  # 좌표계 (수정 가능)
        transform=rasterio.Affine(1, 0, 0, 0, -1, 0)  # 기본 변환 (좌표 없음)
    ) as dst:
        dst.write(mask, 1)

def main():
    image_folder = "Field_Test_Dataset/images"
    mask_folder = "Field_Test_Dataset/labels"
    num_classes = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unetpp_model = load_UNet_model("weights/UNet++_weights/unetpp_resnet50_epoch_50.pth", num_classes, device)
    segformer_model = load_segformer_model("weights/Segformer_weights(b3)/Segformer_epoch_50.pth", num_classes, device)

    image_transform = transforms.Compose([
        #transforms.Lambda(enhance_edges),  # 엣지 보강
        #transforms.Lambda(enhance_saturation),  # 채도 강화
        #transforms.Lambda(enhance_clahe),  # CLAHE 적용
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".tif")])

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)
        output_path = os.path.join(output_folder, image_file)

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        input_image = preprocess_image(image, image_transform).to(device)

        # 모델 예측
        with torch.no_grad():
            output_segformer = segformer_model(input_image)
            prob_segformer = torch.softmax(output_segformer, dim=1).cpu().numpy()
            prob_segformer = prob_segformer.squeeze(0)
            pred_segformer = np.argmax(prob_segformer, axis=0)

            output_unet = unetpp_model(input_image)
            prob_unet = torch.softmax(output_unet, dim=1).cpu().numpy()
            prob_unet = prob_unet.squeeze(0)
            pred_unet = np.argmax(prob_unet, axis=0)

##################################soft voting 앙상블##############################################
        alpha = 0.6
        final_prob = alpha * prob_segformer + (1 - alpha) * prob_unet
        final_prediction = np.argmax(final_prob, axis=0)

        softmax_output = np.expand_dims(final_prob, axis=0)
        prediction = final_prediction

        # ✅ 엣지 기반 후처리 적용
        edge_map = create_edge_map(image, low_threshold=200, high_threshold=400)
        prediction = uncertainty_based_road_refine(prediction, softmax_output, threshold = 0.7)

        refined_prediction = adaptive_road_filling(prediction, softmax_output, base_threshold=0.08, road_class=1)
        refined_prediction = refine_road_with_edges(refined_prediction, softmax_output, edge_map, threshold=0.3,
                                                    road_class=1)

        refined_prediction = directional_road_filter(refined_prediction, road_class=1, min_length=20,
                                                     angle_threshold=30, neighborhood_size=2)

        refined_prediction = contour_simplification(refined_prediction, epsilon=0.9, road_class=1)

        # ✅ Torch Tensor 변환
        refined_prediction = torch.tensor(refined_prediction, dtype = torch.long).cpu()

#####################################################################################################
        save_prediction(refined_prediction, output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
