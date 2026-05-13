# logic/color.py
# ============================================================================
# 【行人版】color.py
# ----------------------------------------------------------------------------
# 修改摘要：
#   1. CLASS_COLORS_RGBA 改為 COCO 80 類對應：
#        - class 0 (person) 給亮綠色
#        - 其他常見類別 (bicycle / car / bus / truck) 保留顏色定義以利未來擴充
#   2. 移除 NUM_MAP 與 LABEL_NUM_FILE：單層版不再做車牌字元辨識
#   3. get_class_color() 簡化 fallback（不再用 modulo，未定義類別直接回白）
#   4. labels 檔內容請替換為 COCO 80 類（檔名保留 labels_car.txt 不變）
# ============================================================================

import os

# ============================================================================
# COCO 80 類別顏色表 (RGBA, 0.0 ~ 1.0)
# 行人版 probes.py 已過濾僅處理 class 0 (person)，
# 其他類別目前不會出現在 OSD，保留定義是為了未來想擴充時直接可用。
# ============================================================================
CLASS_COLORS_RGBA = {
    0: (0.20, 1.00, 0.20, 1.0),   # person     - 亮綠 ⭐ 行人主色
    1: (0.00, 0.80, 0.80, 1.0),   # bicycle    - 青藍
    2: (1.00, 0.00, 0.00, 1.0),   # car        - 紅
    3: (0.00, 0.00, 1.00, 1.0),   # motorbike  - 藍
    5: (0.82, 0.41, 0.12, 1.0),   # bus        - 巧克力
    7: (1.00, 0.39, 0.00, 1.0),   # truck      - 暗橘
}

DEFAULT_COLOR_RGBA = (1.0, 1.0, 1.0, 1.0)   # 白色（未定義類別）


def get_class_color(cls_id: int):
    """取得指定類別的 RGBA 顏色，未定義則回傳白色。"""
    return CLASS_COLORS_RGBA.get(cls_id, DEFAULT_COLOR_RGBA)


def load_labels(label_path):
    """從 labels.txt 載入類別編號到名稱的映射 (line index → class name)"""
    class_map = {}
    if not os.path.exists(label_path):
        print(f"[WARNING] 找不到標籤檔 {label_path}，將使用預設顯示 ID。")
        return class_map
    with open(label_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                class_map[idx] = class_name
    print(f"[INFO] 成功載入 {len(class_map)} 個類別標籤從 {os.path.basename(label_path)}")
    return class_map


# ============================================================================
# 自動初始化 CLASS_MAP
# 檔案路徑保留為 labels_car.txt（避免動到 config.py 的常數定義），
# 但內容請替換為 COCO 80 類。
# ============================================================================
BASE_DIR = "/home/nvidia/DeepStream-Person_Count"
LABEL_CAR_FILE = f"{BASE_DIR}/labels_yolo11s.txt"

CLASS_MAP = load_labels(LABEL_CAR_FILE)

# ⭐ 行人版：已移除 NUM_MAP / LABEL_NUM_FILE（不再做車牌字元辨識）