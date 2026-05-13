# logic/config.py
"""
config.py
---------
集中讀取 ds_yaml/*.yaml 並建立全域 SOURCE_CONFIGS。

每個 cam 的設定欄位：
    - source / stream_fps / start_time
    - device_code                                    ⭐ 行人版新增（=YAML device.code）
    - geometry.regions (多 ROI) / crop_points        ⭐ 行人版改：單 ROI → 多 ROI
    - track_logic.min_roi_hits                       ⭐ 行人版改：移除 movement_threshold
    - session.cleanup_frames / flush_interval_seconds
    - output.save_output_video / output_video_dir / output_excel_dir
    - display.show_window / show_fps_overlay / show_roi / show_crop
    - rtsp_push.enable / port / mount_path / bitrate / encoder
"""

import os
import sys
import glob
import cv2
import yaml
import urllib.parse
import numpy as np

BASE_DIR = "/home/nvidia/DeepStream-Person_Count"
YAML_DIR = f"{BASE_DIR}/ds_yaml"

# Primary 推論與通用設定
INFER_CONFIG = f"{BASE_DIR}/config_infer_primary_yolo11.txt"
TRACKER_CONFIG = f"{BASE_DIR}/config_tracker_NvDCF_accuracy.yml"
PREPROCESS_CONFIG = f"{BASE_DIR}/config_preprocess.txt"
ANALYTICS_CONFIG = f"{BASE_DIR}/config_nvdsanalytics.txt"

# ⭐ 行人版：移除 plate / num 二級推論相關常數（單層偵測，不再有 SGIE）


def _parse_start_time(start_time_str):
    """
    將 YAML 字串 "YYYY-MM-DD HH:MM:SS" 解析成 datetime 物件。
    解析失敗或未提供時回傳 None。
    """
    if not start_time_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(str(start_time_str), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[WARNING] start_time 格式錯誤 ({start_time_str})，將忽略此欄位: {e}")
        return None


def load_dynamic_configs(yaml_dir):
    """
    讀取所有 YAML 設定檔，解析座標、輸出路徑、FPS、裝置代碼與 RTSP 推流設定。

    對每個 yaml 檔，會額外計算/補上下列欄位：
        - cv_regions       : ⭐ 行人版：dict {roi_name: np.int32 array}
                             每個 ROI 的多邊形頂點，給 cv2.pointPolygonTest 用
        - device_code      : ⭐ 行人版：YAML device.code，寫入 CSV 的 DeviceCode 欄位
        - excel_path       : CSV 寫入路徑（依 source_id 命名）
        - video_path       : 輸出影片路徑（依 source_id 命名）
        - is_file_source   : 來源是否為本地檔案（True / False）
        - start_time_dt    : 影片首幀對應的真實時刻（datetime）；非檔案則為 None
        - stream_fps       : 檔案模式會用 cv2 真實抓 FPS 覆寫
        - rtsp_push        : 推流子設定 dict（enable / port / mount_path / bitrate / encoder）
    """
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔於 {yaml_dir}")
        sys.exit(1)

    configs = {}
    for pad_index, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        cam_name = data.get("source_id", f"cam_{pad_index}")

        # ---- 裝置代碼 (CSV DeviceCode 欄位用) ----
        # 結構：device: { code: "EdgeX317" }
        # 若 YAML 沒寫，給合理預設值避免 KeyError
        device_cfg = data.get("device", {}) or {}
        data["device_code"] = str(device_cfg.get("code", "UNKNOWN"))

        # ---- 計數 ROI 轉 numpy（給 probe 的 pointPolygonTest 用）⭐ 行人版改為多 ROI ----
        # YAML 結構：
        #   geometry:
        #     regions:
        #       roi_1: [[x,y], [x,y], ...]
        #       roi_2: [[x,y], [x,y], ...]
        # 解析後：cv_regions = {"roi_1": ndarray, "roi_2": ndarray, ...}
        # 使用 dict 而非 list 是為了保留 roi 名稱（要寫進 CSV ROI 欄位）
        regions_raw = data.get("geometry", {}).get("regions", {}) or {}
        cv_regions = {}
        for roi_name, pts in regions_raw.items():
            if pts and len(pts) >= 3:           # 至少要 3 個點才能形成多邊形
                cv_regions[str(roi_name)] = np.array(pts, np.int32)
            else:
                print(f"[WARNING] {cam_name} 的 ROI '{roi_name}' 點數不足（{len(pts) if pts else 0}），略過")
        data["cv_regions"] = cv_regions
        if not cv_regions:
            print(f"[WARNING] {cam_name} 沒有任何有效的 ROI，將不會產生任何 CSV 紀錄")

        # ---- 輸出路徑（CSV）----
        output_cfg = data.get("output", {})
        excel_dir = output_cfg.get("output_excel_dir", "output_excel")
        os.makedirs(excel_dir, exist_ok=True)
        data["excel_path"] = os.path.join(excel_dir, f"{cam_name}.csv")

        # ---- 輸出路徑（影片）----
        video_dir = output_cfg.get("output_video_dir", "output_video")
        if output_cfg.get("save_output_video", False):
            os.makedirs(video_dir, exist_ok=True)
        data["video_path"] = os.path.join(video_dir, f"{cam_name}_output.mp4")

        # ---- 來源 URI 與 FPS ----
        source_uri = data.get("source", "")
        yaml_fps = data.get("stream_fps", 30.0)

        # RTSP 帳密特殊字元安全編碼
        if source_uri.startswith("rtsp://"):
            try:
                parsed = urllib.parse.urlparse(source_uri)
                if parsed.username and parsed.password:
                    safe_username = urllib.parse.quote(urllib.parse.unquote(parsed.username))
                    safe_password = urllib.parse.quote(urllib.parse.unquote(parsed.password))
                    safe_netloc = f"{safe_username}:{safe_password}@{parsed.hostname}"
                    if parsed.port:
                        safe_netloc += f":{parsed.port}"
                    source_uri = urllib.parse.urlunparse((parsed.scheme, safe_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
                    print(f"[INFO] RTSP URI 安全格式: {source_uri}")
            except Exception as e:
                print(f"[WARNING] 解析 RTSP URI 失敗: {e}")

        # 檔案模式：自動用 cv2 抓真實 FPS 覆寫 YAML 的 stream_fps
        if source_uri.startswith("file://"):
            file_path = source_uri.replace("file://", "")
            cap = cv2.VideoCapture(file_path)
            try:
                if cap.isOpened():
                    real_fps = cap.get(cv2.CAP_PROP_FPS)
                    if real_fps > 0:
                        yaml_fps = real_fps
            finally:
                cap.release()

        data["stream_fps"] = yaml_fps
        data["source"] = source_uri
        data["is_file_source"] = source_uri.startswith("file://")

        # ---- start_time（給 state.py 算「真實事件時刻」用，僅檔案模式生效）----
        data["start_time_dt"] = _parse_start_time(
            data.get("start_time") if data["is_file_source"] else None
        )

        # ---- RTSP 推流設定（給 pipeline.py 與 main.py 用）----
        # 標準化成一個 dict，缺欄位給合理預設
        rtsp_cfg = data.get("rtsp_push", {}) or {}
        data["rtsp_push"] = {
            "enable":     bool(rtsp_cfg.get("enable", False)),
            "port":       int(rtsp_cfg.get("port", 8554)),
            "mount_path": str(rtsp_cfg.get("mount_path", cam_name)),
            "bitrate":    int(rtsp_cfg.get("bitrate", 4000000)),
            "encoder":    str(rtsp_cfg.get("encoder", "h264")).lower(),
        }

        configs[pad_index] = data

    return configs


# 啟動時自動載入配置
SOURCE_CONFIGS = load_dynamic_configs(YAML_DIR)