import numpy as np
import cv2


#엣지 검출
def create_edge_map(image, low_threshold, high_threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges

#####################탐색 판단 알고리즘#######################
def refine_labels(edge_map, label_map, num_classes, class_mapping):
    height, width = edge_map.shape
    refined_label_map = np.copy(label_map)
    DR = 0.7  # Dominative Rate 설정

    # 8방향 탐색을 위한 델타값 설정
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if edge_map[y, x] == 255:  #강한 에지 픽셀인 경우
                current_label = label_map[y, x]
                labels_around = [label_map[y + dy][x + dx] for dy, dx in directions]
                label_counts = np.bincount(labels_around, minlength=num_classes)
                max_label = np.argmax(label_counts)

                # 조건 1: 주변에서 가장 많이 나타난 레이블로 변경
                if max_label != current_label:
                    refined_label_map[y, x] = max_label
                    continue

                # 조건 2: 경계직선화 : DR 적용
                a = np.sum(np.array(labels_around) != current_label)
                b = label_counts[max_label]
                distance = np.sum(label_counts)

                if a < b and b/distance >= DR:
                    refined_label_map[y, x] = max_label
                    continue

                # 조건 3 : 작은 영역 (노이즈 제거)
                if len(label_counts) > 2:
                    second_max_label = np.argsort(label_counts)[-2]
                    if b <= 5 and refined_label_map[y, x] == second_max_label:
                        refined_label_map[y, x] = max_label

                # 예: 연속된 레이블 변화 확인, 레이블 지배도 검사 등
    return refined_label_map
######################################################################################################################
# 소프트 맥스 확률값 적용
def adaptive_road_filling(prediction, softmax_output, base_threshold, road_class):
    refined_prediction = prediction.copy()
    road_prob = softmax_output[0, road_class, :, :]

    #임계값 적용 (도로 중심부,경계부 따로 적용)
    high_confidence_threshold = 0.5  # 중심부
    low_confidence_threshold = base_threshold  # 경계부 (=기본값)

    # 기존 도로 영역 확장
    high_confidence_mask = (road_prob > high_confidence_threshold) & (prediction == road_class)

    # 주변 도로 예측을 확장
    low_confidence_mask = (road_prob > low_confidence_threshold) & (prediction != road_class)

    refined_prediction[high_confidence_mask] = road_class
    refined_prediction[low_confidence_mask] = road_class

    return refined_prediction


# 신뢰도 강화
def refine_road_with_edges(prediction, softmax_output, edge_map, threshold, road_class):
    refined_prediction = prediction.copy()
    road_prob = softmax_output[0, road_class, :, :]

    # 도로 예측값이 threshold 이상이고, 엣지맵에서도 검출된 영역만 보정
    edge_road_mask = (road_prob > threshold) & (edge_map > 0)

    refined_prediction[edge_road_mask] = road_class
    return refined_prediction

########################################################################

# 방향성 고려하여 보정 (PCA 기반)
def directional_road_filter(label_map, road_class, min_length, angle_threshold, neighborhood_size):
    height, width = label_map.shape
    road_mask = (label_map == road_class).astype(np.uint8)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refined_label_map = label_map.copy()
    directions = [(dy, dx) for dy in range(-neighborhood_size, neighborhood_size + 1)
                            for dx in range(-neighborhood_size, neighborhood_size + 1)
                            if not (dy == 0 and dx == 0)]  # 중심 제외

    for contour in contours:
        if len(contour) < min_length:
            continue

        contour_points = contour[:, 0, :]
        mean_x, mean_y = np.mean(contour_points, axis=0)
        cov_matrix = np.cov(contour_points.T)
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        principal_direction = eig_vectors[:, np.argmax(eig_values)]
        road_angle = np.arctan2(principal_direction[1], principal_direction[0]) * (180 / np.pi)

        for point in contour_points:
            x, y = point
            if 0 <= y < height and 0 <= x < width:
                surrounding_labels = []
                surrounding_angles = []

                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        surrounding_labels.append(label_map[ny, nx])
                        surrounding_angles.append(np.arctan2(dy, dx) * (180 / np.pi))

                if surrounding_labels:
                    most_common_label = np.bincount(surrounding_labels).argmax()
                    angle_diffs = np.abs(np.array(surrounding_angles) - road_angle)
                    aligned_count = np.sum(angle_diffs < angle_threshold)

                    # 방향성이 유지되는 경우 도로 유지
                    if aligned_count / len(surrounding_angles) > 0.5:
                        refined_label_map[y, x] = road_class
                    else:
                        refined_label_map[y, x] = most_common_label

    return refined_label_map

#########################################################################################

# 컨투어링 직선화
def contour_simplification(label_map, epsilon, road_class = 1):
    refined_label_map = np.copy(label_map)
    # 도로 클래스만 선택
    road_mask = (label_map == road_class).astype(np.uint8)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    approx_mask = np.zeros_like(road_mask)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)  # 직선화된 컨투어 생성
        cv2.drawContours(approx_mask, [approx], -1, 1, thickness=cv2.FILLED)

    refined_label_map[approx_mask == 1] = road_class

    return refined_label_map

#######################################################################################

# 도로 중점 불확실성 기반 후처리
def uncertainty_based_road_refine(prediction, softmax_output, threshold):
    refined_prediction = prediction.copy()

    # 도로 클래스(1)에서 확신이 낮은 픽셀 찾기
    road_class = 1
    max_confidence = np.max(softmax_output, axis=1)
    uncertainty_mask = (prediction == road_class) & (max_confidence < threshold)
    uncertainty_mask = np.squeeze(uncertainty_mask.astype(bool))

    # 주변 픽셀에서 가장 많이 등장하는 클래스로 보정
    height, width = prediction.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if uncertainty_mask[y, x]:
                surrounding_labels = [prediction[y + dy, x + dx] for dy, dx in directions]
                refined_prediction[y, x] = np.bincount(surrounding_labels).argmax()  # 다수결 결정

    return refined_prediction

###################################################################################################
# 스켈레톤화
def skeletonize_opencv(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    skeleton = np.zeros_like(binary_mask)  # 빈 이미지 생성
    temp = binary_mask.copy()

    # OpenCV_Morphological Thinning 적용
    while True:
        eroded = cv2.erode(temp, np.ones((3, 3), np.uint8))
        temp = temp - eroded
        skeleton = np.maximum(skeleton, temp)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


