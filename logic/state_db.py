# logic/state_db.py
# ============================================================================
# 【行人版】state_db.py - SQLite 即時事件紀錄版
# ----------------------------------------------------------------------------
# 與上一版差異：
#   1. 移除 _finalize_one()：不再「行人消失時才結算」
#   2. 新增 emit_roi_event()：在 ROI 連續命中數「剛達 min_hits」時即時 emit
#   3. 紀錄欄位語意調整：
#        - HitCount  固定 = min_roi_hits (例：45)
#        - VideoTime = 達門檻的影片時間 (frame_num / fps)
#        - CreateTime = 達門檻的真實時間 (start_time + VideoTime)
#   4. 同一 ROI 離開後可再次累積、再次紀錄(無 finalized_rois 鎖)
#   5. force_finalize_all() 簡化為「flush pending + 關閉 DB」
# ============================================================================

import os
import time
import sqlite3
import threading
from datetime import timedelta

from logic.config import SOURCE_CONFIGS
from logic.color import CLASS_MAP

# ============================================================================
# 核心狀態字典(供 probes.py 直接 import 使用)
# ============================================================================
track_history = {}          # key: (pad_index, obj_id) -> 行人軌跡狀態
pending_records = {}        # key: pad_index -> list of tuple (待寫 DB 批次)
last_flush_times = {}       # key: pad_index -> 上次 flush 時間戳
fps_streams = {}            # key: pad_index -> {"current_fps", "timestamps"}
local_id_maps = {}          # key: pad_index -> {global_id: local_id}
next_local_ids = {}         # key: pad_index -> 下一個可用 local_id

# ============================================================================
# SQLite 連線管理
# ============================================================================
_db_conns = {}
_db_lock = threading.Lock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    DeviceCode  TEXT    NOT NULL,
    CameraCode  TEXT    NOT NULL,
    TrackID     INTEGER NOT NULL,
    Class       TEXT,
    ROI         TEXT    NOT NULL,
    HitCount    INTEGER NOT NULL,
    VideoTime   TEXT,
    CreateTime  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_camera_time
    ON events (CameraCode, CreateTime);

CREATE INDEX IF NOT EXISTS idx_roi
    ON events (ROI);
"""


def _get_db_path(cfg, pad_index):
    """從 cfg["excel_path"] 推算 DB 路徑(把 .csv 換成 .db)。"""
    excel_path = cfg.get("excel_path", f"output_excel/cam_{pad_index}.csv")
    base, _ = os.path.splitext(excel_path)
    return f"{base}.db"


def _open_db(pad_index, cfg):
    """為某路 cam 開啟 SQLite connection 並建立 schema。"""
    db_path = _get_db_path(cfg, pad_index)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)
    print(f"[INFO] SQLite DB 開啟: {db_path}")
    return conn


# ============================================================================
# Public API
# ============================================================================

def initialize_state_managers():
    """為每一個 cam 初始化狀態管理器與 DB 連線。"""
    for pad_index, cfg in SOURCE_CONFIGS.items():
        pending_records[pad_index] = []
        last_flush_times[pad_index] = time.time()
        fps_streams[pad_index] = {"current_fps": 0.0}
        local_id_maps[pad_index] = {}
        next_local_ids[pad_index] = 1
        _db_conns[pad_index] = _open_db(pad_index, cfg)


def get_local_id(pad_index, global_id):
    """全局 ID → 該路的短 ID(1,2,3...)。"""
    if global_id not in local_id_maps[pad_index]:
        local_id_maps[pad_index][global_id] = next_local_ids[pad_index]
        next_local_ids[pad_index] += 1
    return local_id_maps[pad_index][global_id]


def _format_video_time(vsec):
    """秒數 → HH:MM:SS。"""
    if vsec is None or vsec < 0:
        return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(int(vsec)))


def emit_roi_event(pad_index, local_id, class_id, roi_name, frame_num, hit_count):
    """
    在 ROI 連續命中數「剛達 min_hits」的當下呼叫，產生一筆事件紀錄。

    Args:
        pad_index   : 第幾路 cam
        local_id    : 該路內的短 ID
        class_id    : 觸發當下物件類別(行人版固定為 0=person)
        roi_name    : 觸發的 ROI 名稱
        frame_num   : 觸發當下的影片幀號(用於算 VideoTime / CreateTime)
        hit_count   : 寫入 DB 的命中數(=min_roi_hits，固定值)

    紀錄會放入 pending_records，等 flush_pending_to_db 批次寫入。
    """
    cfg = SOURCE_CONFIGS.get(pad_index, {})

    device_code = cfg.get("device_code", "UNKNOWN")
    camera_code = cfg.get("source_id", f"cam_{pad_index}")
    cls_name = CLASS_MAP.get(class_id, f"Class_{class_id}")

    # VideoTime = 觸發當下的影片內時間
    vsec = frame_num / cfg.get("stream_fps", 30.0)
    time_axis = _format_video_time(vsec)

    # CreateTime = 觸發當下的真實時間(file: start_time + vsec；live: now())
    start_dt = cfg.get("start_time_dt")
    if start_dt is not None:
        event_dt = start_dt + timedelta(seconds=vsec)
        create_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    pending_records[pad_index].append((
        device_code,
        camera_code,
        local_id,
        cls_name,
        roi_name,
        hit_count,
        time_axis,
        create_time_str,
    ))

    print(f"[ROI觸發][{camera_code}] TrackID={local_id}, 類別={cls_name}, "
          f"ROI={roi_name}, 次數={hit_count}, "
          f"VideoTime={time_axis}, CreateTime={create_time_str}")


def flush_pending_to_db(pad_index):
    """把 pending_records[pad_index] 批次寫入 SQLite DB，回傳實際寫入筆數。"""
    records = pending_records.get(pad_index, [])
    if not records:
        return 0

    conn = _db_conns.get(pad_index)
    if conn is None:
        print(f"[WARNING] pad_index={pad_index} 沒有 DB 連線，丟棄 {len(records)} 筆紀錄")
        records.clear()
        return 0

    with _db_lock:
        try:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO events "
                "(DeviceCode, CameraCode, TrackID, Class, ROI, HitCount, VideoTime, CreateTime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            conn.execute("COMMIT")
            n = len(records)
            records.clear()
            return n
        except sqlite3.Error as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            print(f"[ERROR] SQLite 寫入失敗 (pad_index={pad_index}): {e}")
            return 0


def force_finalize_all():
    """
    程式結束前呼叫：把 pending buffer 的剩餘紀錄寫入 DB，然後關閉所有 DB。

    ⭐ 行人版即時 emit：不再做「未達門檻強制結算」的事，因為紀錄是
       「達門檻當下立即 emit」的，未達門檻的軌跡本來就不應該寫進 DB。
    """
    print("\n[INFO] 開始執行強制結算(僅 flush 剩餘 pending)...")

    for pad_index, cfg in SOURCE_CONFIGS.items():
        n = flush_pending_to_db(pad_index)
        if n > 0:
            db_path = _get_db_path(cfg, pad_index)
            print(f"[檔案儲存] {cfg.get('source_id')}：已強制寫入 {n} 筆剩餘資料到 {db_path}")

    for pad_index, conn in list(_db_conns.items()):
        try:
            conn.close()
        except Exception as e:
            print(f"[WARNING] 關閉 DB 連線失敗 (pad_index={pad_index}): {e}")
    _db_conns.clear()

    track_history.clear()