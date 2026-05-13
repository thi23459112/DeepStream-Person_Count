# logic/state.py
# ============================================================================
# 【行人版】state.py
# ----------------------------------------------------------------------------
# 修改摘要：
#   1. CSV 欄位（依需求重新定義）：
#        DeviceCode / CameraCode / TrackID / 類別 / ROI / 次數 / 時間軸 / CreateTime
#   2. 多 ROI 結算：
#        - state["roi_hits"] 是 dict {roi_name: count}
#        - 對「每個達門檻的 ROI」各產一筆紀錄（一個行人可能有多筆）
#        - 使用 finalized_rois 集合避免同一 ROI 重複輸出
#   3. 終端 print 訊息加 ROI=roi_name
# ============================================================================

import os
import time
import pandas as pd

from logic.config import SOURCE_CONFIGS
from logic.color import CLASS_MAP

# ============================================================================
# 核心狀態字典（全局變數）
# ============================================================================
track_history = {}          # key: (pad_index, obj_id) -> 行人軌跡狀態字典
pending_records = {}        # key: pad_index -> list of 待寫入 CSV 的記錄
last_flush_times = {}       # key: pad_index -> 上次寫入 CSV 的時間戳
fps_streams = {}            # key: pad_index -> {"current_fps": 0.0, "timestamps": deque}
local_id_maps = {}          # key: pad_index -> dict {global_id: local_id}
next_local_ids = {}         # key: pad_index -> 下一個可用的 local_id


def initialize_state_managers():
    """為每一個 cam (pad_index) 初始化狀態管理器。必須在 pipeline 建立前呼叫一次。"""
    for pad_index in SOURCE_CONFIGS.keys():
        pending_records[pad_index] = []
        last_flush_times[pad_index] = time.time()
        fps_streams[pad_index] = {"current_fps": 0.0}
        local_id_maps[pad_index] = {}
        next_local_ids[pad_index] = 1


def get_local_id(pad_index, global_id):
    """全局 ID → 該路的短 ID（1,2,3...），給 OSD 顯示與 CSV TrackID 用。"""
    if global_id not in local_id_maps[pad_index]:
        local_id_maps[pad_index][global_id] = next_local_ids[pad_index]
        next_local_ids[pad_index] += 1
    return local_id_maps[pad_index][global_id]


def _format_video_time(vsec: float) -> str:
    """秒數 → HH:MM:SS。"""
    if vsec is None or vsec < 0:
        return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(int(vsec)))


def _finalize_one(m_key, state, force=False):
    """
    結算單一行人的多 ROI 統計。

    【行人版多 ROI 邏輯】：
        - 遍歷 state["roi_hits"] = {roi_name: count}
        - 對每個 count >= min_roi_hits 且尚未結算過的 ROI，各產生一筆紀錄
        - 用 state["finalized_rois"] 記錄已結算過的 ROI，避免同一 ROI 重複出
          （主要在 force_finalize_all 與 cleanup_frames 同時觸發時防呆）
    """
    pad_index, obj_id = m_key
    local_id = get_local_id(pad_index, obj_id)
    cfg = SOURCE_CONFIGS.get(pad_index, {})

    # CSV 欄位來源
    device_code = cfg.get("device_code", "UNKNOWN")
    camera_code = cfg.get("source_id", f"cam_{pad_index}")
    min_hits = cfg.get("track_logic", {}).get("min_roi_hits", 45)

    # ---------- 類別投票（行人版幾乎固定為 person）----------
    if state["class_votes"]:
        best_class_id = state["class_votes"].most_common(1)[0][0]
        cls_name = CLASS_MAP.get(best_class_id, f"Class_{best_class_id}")
    else:
        cls_name = "Unknown"

    # ---------- 時間軸（影片內偏移秒數）----------
    vsec = state["last_frame_num"] / cfg.get("stream_fps", 30.0)
    time_axis = _format_video_time(vsec)

    # ---------- 真實事件時間戳記（CreateTime 欄位）----------
    start_dt = cfg.get("start_time_dt")
    if start_dt is not None:
        from datetime import timedelta
        event_dt = start_dt + timedelta(seconds=vsec)
        create_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 對每個達門檻的 ROI 各產生一筆紀錄 ----------
    finalized = state.setdefault("finalized_rois", set())
    roi_hits = state.get("roi_hits", {})

    for roi_name, hits in roi_hits.items():
        if hits < min_hits:
            continue
        if roi_name in finalized:
            continue

        pending_records[pad_index].append({
            "DeviceCode": device_code,
            "CameraCode": camera_code,
            "TrackID":    local_id,
            "DetectClass":cls_name,
            "ROI":        roi_name,
            "次數":        hits,
            "時間軸":      time_axis,
            "RecordTime": create_time_str,
        })
        finalized.add(roi_name)

        tag = "[統計結算-強制]" if force else "[統計結算]"
        print(f"{tag}[{camera_code}] TrackID={local_id}, 類別={cls_name}, "
              f"ROI={roi_name}, 次數={hits}, CreateTime={create_time_str}")


def force_finalize_all():
    """
    強制結算所有尚未完成的行人，並把 buffer 內剩餘紀錄寫入 CSV。
    通常在程式結束前呼叫。
    """
    print("\n[INFO] 開始執行強制結算...")

    for m_key, state in list(track_history.items()):
        _finalize_one(m_key, state, force=True)

    for pad_index, cfg in SOURCE_CONFIGS.items():
        records = pending_records[pad_index]
        if records:
            excel_path = cfg["excel_path"]
            pd.DataFrame(records).to_csv(excel_path, mode='a',
                                         header=not os.path.exists(excel_path),
                                         index=False, encoding='utf-8-sig')
            print(f"[檔案儲存] {cfg.get('source_id')}：已強制寫入 {len(records)} 筆剩餘資料到 {excel_path}")
            records.clear()

    track_history.clear()