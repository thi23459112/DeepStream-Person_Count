#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
 DeepStream 設定檔自動產生器(行人版)
================================================================================

讀取 ds_yaml/*.yaml，自動產生以下四個 DeepStream 組態檔：
    1. deepstream_app_config.txt            主應用設定
    2. config_preprocess.txt                前處理(裁切 ROI、縮放)
    3. config_infer_primary_yolo11.txt      PGIE 行人偵測(yolo11s_fp16.engine)
    4. config_nvdsanalytics.txt             ROI 區域繪製(支援多 ROI)

================================================================================
 從 YAML 讀進來的關鍵欄位
================================================================================
    weight_imgsz              : 模型輸入解析度
    weight_batch_size         : engine 的 max batch
    detect.person_conf/iou    : PGIE 信心值與 NMS IoU 閾值
    geometry.base_w/h         : 各路畫面原始解析度
    geometry.crop_points      : 裁切 ROI(preprocess 用)
    geometry.regions          : 多個計數 ROI(nvdsanalytics 用)⭐ 多 ROI
    display.show_roi          : 是否在畫面上畫 ROI 黃線
    display.show_crop         : 是否在畫面上畫裁切框
    source                    : 影片或串流 URI
================================================================================
"""

import os
import sys
import glob
import yaml
import math
from typing import List, Dict, Any, Tuple

# ============================================================================
# 路徑設定
# ============================================================================
BASE_DIR = "/home/nvidia/DeepStream-Person_Count"
YAML_DIR = f"{BASE_DIR}/ds_yaml"

LABEL_FILE = "labels_yolo11s.txt"
ENGINE_FILE = "yolo11s_fp16.engine"

# ============================================================================
# 預設值
# ============================================================================
DEFAULT_WEIGHT_IMGSZ = 640
DEFAULT_WEIGHT_BATCH = 4

# ============================================================================
# 產生的設定檔絕對路徑
# ============================================================================
APP_CONFIG           = f"{BASE_DIR}/deepstream_app_config.txt"
PREPROCESS_CONFIG    = f"{BASE_DIR}/config_preprocess.txt"
INFER_PRIMARY_CONFIG = f"{BASE_DIR}/config_infer_primary_yolo11.txt"
ANALYTICS_CONFIG     = f"{BASE_DIR}/config_nvdsanalytics.txt"


# ============================================================================
# 通用輔助函式
# ============================================================================

def load_all_yamls(yaml_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔案在：{yaml_dir}")
        sys.exit(1)

    cfgs = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            cfgs.append(data)
    return cfgs


def get_num_classes(label_filename: str) -> int:
    """讀 labels 檔的非空行數，COCO 80 類會得到 80。"""
    label_path = os.path.join(BASE_DIR, label_filename)
    if not os.path.exists(label_path):
        print(f"[WARNING] 標籤檔案不存在：{label_path}，使用預設值 1")
        return 1

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        print(f"[WARNING] 標籤檔案 {label_path} 內容為空，使用預設值 1")
        return 1
    return len(lines)


def get_detect_thresholds(cfgs: List[Dict[str, Any]]) -> Tuple[float, float]:
    """讀 detect.person_conf / person_iou。"""
    detect = cfgs[0].get("detect", {})
    return detect.get("person_conf", 0.25), detect.get("person_iou", 0.45)


def get_engine_batch(cfgs: List[Dict[str, Any]]) -> int:
    return int(cfgs[0].get("weight_batch_size", DEFAULT_WEIGHT_BATCH))


def get_engine_imgsz(cfgs: List[Dict[str, Any]]) -> int:
    return int(cfgs[0].get("weight_imgsz", DEFAULT_WEIGHT_IMGSZ))


def crop_points_to_rect(points: List[List[int]]) -> Tuple[int, int, int, int]:
    """多邊形點位 → 外接矩形 (x, y, w, h)。preprocess 用。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(1, int(max_x - min_x))
    height = max(1, int(max_y - min_y))
    return int(min_x), int(min_y), width, height


def resolve_muxer_size(cfgs: List[Dict[str, Any]]) -> Tuple[int, int]:
    max_w = max(cfg["geometry"]["base_w"] for cfg in cfgs)
    max_h = max(cfg["geometry"]["base_h"] for cfg in cfgs)
    return max_w, max_h


def compute_tiled_layout(num_sources: int) -> Tuple[int, int]:
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)
    return rows, cols


def is_live_stream(uri: str) -> bool:
    return uri.startswith(("rtsp://", "http://", "https://"))


def _polygon_to_pts_string(points: List[List[int]]) -> str:
    """
    把多邊形點位 [[x1,y1], [x2,y2], ...] 攤平成
    nvdsanalytics 要的格式 "x1;y1;x2;y2;..."
    """
    return ";".join(str(coord) for point in points for coord in point)


# ============================================================================
# 設定檔產生：preprocess
# ============================================================================

def generate_preprocess_config(cfgs: List[Dict[str, Any]]) -> None:
    camera_count = len(cfgs)
    engine_batch = get_engine_batch(cfgs)
    imgsz = get_engine_imgsz(cfgs)
    show_any_crop = any(cfg.get("display", {}).get("show_crop", False) for cfg in cfgs)

    lines = [
        "[property]",
        "enable=1",
        "target-unique-ids=1",
        "process-on-frame=1",
        "network-input-order=0",
        f"network-input-shape={engine_batch};3;{imgsz};{imgsz}",
        "network-color-format=0",
        "tensor-data-type=0",
        "tensor-name=input",
        f"processing-width={imgsz}",
        f"processing-height={imgsz}",
        "scaling-buf-pool-size=6",
        "tensor-buf-pool-size=6",
        "scaling-pool-memory-type=0",
        "scaling-pool-compute-hw=0",
        "scaling-filter=0",
        "maintain-aspect-ratio=1",
        "symmetric-padding=1",
        "custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so",
        "custom-tensor-preparation-function=CustomTensorPreparation",
        "",
        "[user-configs]",
        "pixel-normalization-factor=0.003921568",
        "",
        "[group-0]",
        f"src-ids={';'.join(str(i) for i in range(camera_count))}",
        "process-on-roi=1",
        "custom-input-transformation-function=CustomAsyncTransformation",
        f"draw-roi={1 if show_any_crop else 0}",
        "roi-color=0;1;1;1",
    ]

    for i, cfg in enumerate(cfgs):
        crop_points = cfg["geometry"]["crop_points"]
        x, y, w, h = crop_points_to_rect(crop_points)
        lines.append(f"roi-params-src-{i}={x};{y};{w};{h}")

    with open(PREPROCESS_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# 設定檔產生：PGIE(行人偵測，per-class 過濾)
# ============================================================================

def generate_primary_infer_config(cfgs: List[Dict[str, Any]]) -> None:
    """
    Per-class 過濾(在 PGIE 層級就只保留 person)：
        - [class-attrs-all] threshold=1.1 → 所有類別預設全砍
        - [class-attrs-0] 蓋掉 class 0 (person) 為正常 conf 閾值
    """
    batch_size = get_engine_batch(cfgs)
    num_classes = get_num_classes(LABEL_FILE)
    conf_thresh, iou_thresh = get_detect_thresholds(cfgs)

    content = f"""\
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file={ENGINE_FILE}
batch-size={batch_size}
network-mode=2
num-detected-classes={num_classes}
labelfile-path={LABEL_FILE}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
parse-bbox-func-name=NvDsInferParseYolo

# 所有類別預設關閉(threshold=1.1 不可能達成 → 全砍)
[class-attrs-all]
pre-cluster-threshold=1.1
nms-iou-threshold={iou_thresh}
topk=300

# 僅啟用 class 0 (person) — COCO 索引
[class-attrs-0]
pre-cluster-threshold={conf_thresh}
nms-iou-threshold={iou_thresh}
topk=300
"""
    with open(INFER_PRIMARY_CONFIG, "w", encoding="utf-8") as f:
        f.write(content)


# ============================================================================
# 設定檔產生：nvdsanalytics(多 ROI 版本)
# ============================================================================

def generate_analytics_config(cfgs: List[Dict[str, Any]], muxer_w: int, muxer_h: int) -> None:
    """
    產生 config_nvdsanalytics.txt(行人版，多 ROI 支援)

    ⭐ 多 ROI 機制：
        - YAML geometry.regions 是一個 dict：{roi_1: [...], roi_2: [...]}
        - 同一個 [roi-filtering-stream-N] 區塊內可放多個 roi-XXX=... 鍵
        - 每個鍵代表一個獨立的多邊形 ROI，nvdsanalytics 會各自畫線
        - OSD 上會顯示每個 ROI 的名稱(roi_1 / roi_2 / 自訂名稱皆可)
          這個名稱應與 probes.py 寫入 CSV 的 ROI 欄位完全一致

    Fallback：若某路 cam 沒設定 regions(或為空)，會用全畫面當作預設 ROI，
              鍵名沿用 source_id，避免 nvdsanalytics 啟動報錯。

    show_roi=False 時整個區塊 enable=0(但仍要產出，DS 要求 stream id 連續)。
    """
    lines = [
        "[property]",
        "enable=1",
        f"config-width={muxer_w}",
        f"config-height={muxer_h}",
        "osd-mode=1",
        "display-font-size=12",
        ""
    ]

    for i, cfg in enumerate(cfgs):
        source_id = cfg.get("source_id", f"cam_{i}")
        show_roi = 1 if cfg.get("display", {}).get("show_roi", True) else 0
        regions = cfg.get("geometry", {}).get("regions", {}) or {}

        block = [
            f"[roi-filtering-stream-{i}]",
            f"enable={show_roi}",
        ]

        if regions:
            # ⭐ 多 ROI：每個 region 輸出一個 roi-{name}=... 鍵
            for roi_name, pts in regions.items():
                if not pts or len(pts) < 3:
                    print(f"[WARNING] {source_id} 的 ROI '{roi_name}' 點數不足({len(pts) if pts else 0})，略過")
                    continue
                pts_str = _polygon_to_pts_string(pts)
                block.append(f"roi-{roi_name}={pts_str}")
        else:
            # Fallback：沒設定就用全畫面當預設 ROI
            print(f"[WARNING] {source_id} 沒有定義 regions，使用全畫面作為預設 ROI")
            pts_str = f"0;0;{muxer_w};0;{muxer_w};{muxer_h};0;{muxer_h}"
            block.append(f"roi-{source_id}={pts_str}")

        block.extend([
            "class-id=-1",
            "inverse-roi=0",
            ""
        ])

        lines.extend(block)

    with open(ANALYTICS_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# 設定檔產生：deepstream_app_config
# ============================================================================

def generate_deepstream_app_config(cfgs: List[Dict[str, Any]], muxer_w: int, muxer_h: int) -> None:
    num_sources = len(cfgs)
    rows, cols = compute_tiled_layout(num_sources)

    display_width = 1280
    display_height = 720

    live_source = 0
    for cfg in cfgs:
        if is_live_stream(cfg.get("source", "")):
            live_source = 1
            break

    lines = [
        "[application]",
        "enable-perf-measurement=1",
        "perf-measurement-interval-sec=1",
        "",
        "[tiled-display]",
        "enable=1",
        f"rows={rows}",
        f"columns={cols}",
        f"width={display_width}",
        f"height={display_height}",
        "gpu-id=0",
        "nvbuf-memory-type=0",
        ""
    ]

    for i, cfg in enumerate(cfgs):
        uri = cfg.get("source", f"file://{BASE_DIR}/videos/source_{i}.mp4")
        if not uri.startswith(("file://", "rtsp://", "http://", "https://")):
            uri = f"file://{uri}"

        source_type = 4 if is_live_stream(uri) else 3

        lines.extend([
            f"[source{i}]",
            "enable=1",
            f"type={source_type}",
            f"uri={uri}",
            "num-sources=1",
            "gpu-id=0",
            "cudadec-memtype=0",
            ""
        ])

    lines.extend([
        "[sink0]",
        "enable=1",
        "type=2",
        "sync=0",
        "qos=0",
        "gpu-id=0",
        "nvbuf-memory-type=0",
        "",
        "[osd]",
        "enable=1",
        "gpu-id=0",
        "border-width=2",
        "text-size=15",
        "text-color=1;1;1;1",
        "text-bg-color=0;0;0;1",
        "font=Serif",
        "display-text=1",
        "display-bbox=1",
        "",
        "[streammux]",
        "gpu-id=0",
        f"live-source={live_source}",
        f"batch-size={num_sources}",
        "batched-push-timeout=40000",
        f"width={muxer_w}",
        f"height={muxer_h}",
        "enable-padding=0",
        "nvbuf-memory-type=0",
        "",
        "[pre-process]",
        "enable=1",
        f"config-file={PREPROCESS_CONFIG}",
        "",
        "[primary-gie]",
        "enable=1",
        "gpu-id=0",
        "gie-unique-id=1",
        "nvbuf-memory-type=0",
        f"config-file={INFER_PRIMARY_CONFIG}",
        "input-tensor-meta=1",
        "",
        "[tracker]",
        "enable=1",
        "tracker-width=640",
        "tracker-height=384",
        "ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        f"ll-config-file={BASE_DIR}/config_tracker_NvDCF_accuracy.yml",
        "gpu-id=0",
        "display-tracking-id=1",
        "",
        "[nvds-analytics]",
        "enable=1",
        f"config-file={ANALYTICS_CONFIG}",
        "",
        "[tests]",
        "file-loop=0"
    ])

    with open(APP_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# 主程式
# ============================================================================

def main() -> None:
    print("正在載入 YAML 設定檔...")
    cfgs = load_all_yamls(YAML_DIR)
    print(f"已載入 {len(cfgs)} 個攝影機設定")

    muxer_w, muxer_h = resolve_muxer_size(cfgs)
    print(f"Streammux 輸出尺寸: {muxer_w} x {muxer_h}")

    eng_batch = get_engine_batch(cfgs)
    eng_imgsz = get_engine_imgsz(cfgs)
    print(f"Engine 設定 — engine={ENGINE_FILE}, batch={eng_batch}, imgsz={eng_imgsz}")

    # 印出每路 cam 的 ROI 設定，方便檢查
    for i, cfg in enumerate(cfgs):
        source_id = cfg.get("source_id", f"cam_{i}")
        regions = cfg.get("geometry", {}).get("regions", {}) or {}
        if regions:
            roi_names = list(regions.keys())
            print(f"  [{source_id}] ROI 數量: {len(roi_names)} → {roi_names}")
        else:
            print(f"  [{source_id}] 未定義 ROI，將使用全畫面")

    generate_preprocess_config(cfgs)
    generate_primary_infer_config(cfgs)
    generate_analytics_config(cfgs, muxer_w, muxer_h)
    generate_deepstream_app_config(cfgs, muxer_w, muxer_h)

    print("\n[DONE] 所有設定檔產生完畢！")
    print(f"  - {APP_CONFIG}")
    print(f"  - {PREPROCESS_CONFIG}")
    print(f"  - {INFER_PRIMARY_CONFIG}")
    print(f"  - {ANALYTICS_CONFIG}")


if __name__ == "__main__":
    main()