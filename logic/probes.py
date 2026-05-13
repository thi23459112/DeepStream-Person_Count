# logic/probes.py
# ============================================================================
# 【行人版】probes.py - 即時 emit 版
# ----------------------------------------------------------------------------
# 與上一版差異：
#   1. ROI 累積邏輯改成「連續累積」：離開 ROI 立即將該 ROI 的 hits 歸零
#   2. 紀錄時機：在 hits 剛達 min_hits 那一幀就呼叫 emit_roi_event 寫入
#   3. 同一 ROI 離開後再進入、再次累積滿，會再 emit 一筆(可重複觸發)
#   4. track_history 不再有 finalized_rois / class_votes 欄位
#   5. 行人消失只清 track_history，不再做任何結算(紀錄已即時寫入)
#   6. 紅框條件不變：當下任一 ROI 命中 >= min_hits → 紅
#      由於 hits 會在離開時歸零，紅框會自然消失
# ============================================================================

import time
import cv2
from collections import deque
from gi.repository import Gst
import pyds

from logic.color import get_class_color, CLASS_MAP
from logic.config import SOURCE_CONFIGS
from logic.state_db import (
    get_local_id, emit_roi_event, flush_pending_to_db,
    track_history, pending_records, last_flush_times,
    fps_streams, local_id_maps
)

g_last_fps_print_time = time.time()

# ⭐ 行人版：只計數 COCO class 0 (person)
PERSON_CLASS_ID = 0

# ⭐ 已達停留門檻時的 bbox 顏色(紅色)
ALERT_COLOR_RGBA = (1.0, 0.0, 0.0, 1.0)


# ==========================================
# 探針 1：追蹤器與行人狀態更新
# ==========================================
def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    掛在 tracker.src pad 上的 buffer probe。

    工作：
      - 維護 track_history，每個 ROI 「連續累積」命中數
      - 離開 ROI 立即將該 ROI 計數歸零
      - 累積數剛達 min_hits 的那一幀 → 即時 emit 一筆事件
      - OSD：當下任一 ROI 累積數 >= min_hits 的行人 bbox 變紅
      - 定期把 pending_records flush 到 SQLite DB
    """
    global g_last_fps_print_time

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        cv_regions = cfg.get("cv_regions", {})
        min_hits = cfg.get("track_logic", {}).get("min_roi_hits", 45)

        # ----- FPS 統計 -----
        if "timestamps" not in fps_streams[pad_index]:
            fps_streams[pad_index]["timestamps"] = deque(maxlen=30)
        now = time.time()
        q = fps_streams[pad_index]["timestamps"]
        q.append(now)
        if len(q) > 1:
            fps_streams[pad_index]["current_fps"] = (len(q) - 1) / (q[-1] - q[0])

        # ----- 遍歷物件 -----
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # 過濾 1：只處理追蹤器輸出
            if obj_meta.unique_component_id != 1:
                l_obj = l_obj.next
                continue

            # 過濾 2：只處理 person
            if obj_meta.class_id != PERSON_CLASS_ID:
                l_obj = l_obj.next
                continue

            obj_id = obj_meta.object_id
            if obj_id == -1:
                l_obj = l_obj.next
                continue

            unique_key = (pad_index, obj_id)
            current_frame_objects.add(unique_key)
            local_id = get_local_id(pad_index, obj_id)

            # bbox 底部中心點
            cx = int(obj_meta.rect_params.left + (obj_meta.rect_params.width / 2))
            cy = int(obj_meta.rect_params.top + obj_meta.rect_params.height)

            # ----- 初始化軌跡狀態 -----
            # roi_hits      : {roi_name: 連續命中數}，離開 ROI 會歸零
            # roi_triggered : {roi_name: 本次連續累積是否已 emit 過}
            #                 防止同一次累積在 hits=45, 46, 47... 重複 emit
            #                 (離開 ROI 時會跟 hits 一起歸零，允許下次再 emit)
            if unique_key not in track_history:
                track_history[unique_key] = {
                    "roi_hits": {},
                    "roi_triggered": {},
                    "missing_frames": 0,
                    "last_frame_num": frame_meta.frame_num,
                }

            state = track_history[unique_key]
            state["missing_frames"] = 0
            state["last_frame_num"] = frame_meta.frame_num

            # ----- 多 ROI 命中判斷(含離開歸零邏輯) -----
            for roi_name, polygon in cv_regions.items():
                inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0

                if inside:
                    # 在 ROI 內：累積
                    state["roi_hits"][roi_name] = state["roi_hits"].get(roi_name, 0) + 1

                    # 剛好達到門檻 → 立即 emit 一筆紀錄
                    # 用 == 而非 >= 確保只在「跨越門檻那一幀」觸發
                    # 配合 roi_triggered 雙保險
                    if (state["roi_hits"][roi_name] >= min_hits
                            and not state["roi_triggered"].get(roi_name, False)):
                        emit_roi_event(
                            pad_index=pad_index,
                            local_id=local_id,
                            class_id=obj_meta.class_id,
                            roi_name=roi_name,
                            frame_num=frame_meta.frame_num,
                            hit_count=min_hits,    # ⭐ 固定寫入 min_hits(=45)
                        )
                        state["roi_triggered"][roi_name] = True
                else:
                    # 離開 ROI：歸零該 ROI 的累積與觸發旗標
                    # 下次再進入可重新累積，累積滿了會再 emit 一筆
                    if state["roi_hits"].get(roi_name, 0) > 0:
                        state["roi_hits"][roi_name] = 0
                        state["roi_triggered"][roi_name] = False

            # ----- 視覺化：當下任一 ROI 連續累積 >= min_hits → 紅框 -----
            cls_id = obj_meta.class_id
            cls_name = CLASS_MAP.get(cls_id, f"Class_{cls_id}")

            triggered = any(c >= min_hits for c in state["roi_hits"].values())
            color = ALERT_COLOR_RGBA if triggered else get_class_color(cls_id)

            r = obj_meta.rect_params
            r.border_width = 4
            r.border_color.set(*color)
            r.has_bg_color = 0

            txt = obj_meta.text_params
            txt.display_text = f"ID:{local_id} {cls_name}"
            txt.font_params.font_name = "Serif Bold"
            txt.font_params.font_size = 14
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(*color)

            text_h = int(14 * 1.4)
            txt.x_offset = max(0, int(r.left) + 0)
            txt.y_offset = max(0, int(r.top + r.height) - text_h - 10)

            l_obj = l_obj.next
        l_frame = l_frame.next

    # ----- 清理消失的行人(只清狀態，不再做結算) -----
    # 紀錄已在 emit 時即時寫入，行人離開畫面後不再有任何 DB 寫入動作
    missing_keys = set(track_history.keys()) - current_frame_objects
    for m_key in missing_keys:
        pad_index, obj_id = m_key
        cfg = SOURCE_CONFIGS.get(pad_index, {})
        track_history[m_key]["missing_frames"] += 1
        cleanup_frames = cfg.get("session", {}).get("cleanup_frames", 90)
        if track_history[m_key]["missing_frames"] >= cleanup_frames:
            del track_history[m_key]
            if obj_id in local_id_maps[pad_index]:
                del local_id_maps[pad_index][obj_id]

    # ----- 每 30 秒印 FPS -----
    current_time = time.time()
    if current_time - g_last_fps_print_time >= 30:
        print("\n" + "=" * 35)
        print(f"[{time.strftime('%H:%M:%S')}] 即時處理效能報告 (FPS)：")
        for sid, stats in sorted(fps_streams.items()):
            c_name = SOURCE_CONFIGS[sid].get("source_id", f"cam_{sid}")
            print(f" • {c_name.ljust(10)}: {stats['current_fps']:.2f} FPS")
        print("=" * 35 + "\n")
        g_last_fps_print_time = current_time

    # ----- 定期 flush 到 SQLite DB -----
    for pad_index, cfg in SOURCE_CONFIGS.items():
        flush_interval = cfg.get("session", {}).get("flush_interval_seconds", 30)
        if current_time - last_flush_times[pad_index] >= flush_interval:
            flush_pending_to_db(pad_index)
            last_flush_times[pad_index] = current_time

    return Gst.PadProbeReturn.OK


# ==========================================
# 探針 2：每路畫面的 FPS OSD
# ==========================================
def per_cam_osd_probe(pad, info, pad_index):
    """掛在每路 nvosd.sink pad 上，於畫面左上角畫即時 FPS 文字。"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    cfg = SOURCE_CONFIGS.get(pad_index)
    if not cfg:
        return Gst.PadProbeReturn.OK

    show_fps = cfg.get("display", {}).get("show_fps_overlay", True)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 0
        display_meta.num_lines = 0
        display_meta.num_rects = 0
        display_meta.num_circles = 0

        if show_fps and pad_index in fps_streams:
            display_meta.num_labels = 1
            txt_params = display_meta.text_params[0]
            txt_params.display_text = f"FPS: {fps_streams[pad_index]['current_fps']:.1f}"
            txt_params.x_offset = 5
            txt_params.y_offset = 5
            txt_params.font_params.font_name = "Serif Bold"
            txt_params.font_params.font_size = 25
            txt_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK