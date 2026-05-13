#!/usr/bin/env python3
"""
main.py
-------
DeepStream 7.1 車流追蹤 主程式（單層版，只跑 car_fp16.engine）。

============================================================================
【車流單層版】修改摘要
----------------------------------------------------------------------------
相較於原本三層版 PGIE→SGIE_plate→SGIE_num，本檔案的差異：
  1. import 區拿掉：
       - INFER_SEC_PLATE_CONFIG / INFER_SEC_NUM_CONFIG（不再載入二級推論設定）
       - expand_plate_probe / assemble_plate_probe（兩支探針已不存在）
  2. 元件建立移除：
       - q_sgie_plate / q_sgie_num（兩個 SGIE 前的 queue）
       - sgie_plate / sgie_num（兩個 nvinfer 二級推論元件）
  3. Pipeline 連結改為：
       tracker → q_analytics → analytics → q4 → demux
       （原本 tracker 後面接 q_sgie_plate→sgie_plate→q_sgie_num→sgie_num→q_analytics）
  4. Probe 掛載移除：
       - sgie_plate.src 上的 expand_plate_probe
       - sgie_num.src   上的 assemble_plate_probe
     僅保留 tracker.src 上的 tracker_src_pad_buffer_probe。
其他（streammux / preprocess / pgie / tracker / analytics / RTSP / 鍵盤監聽
等）邏輯完全保留。
============================================================================

啟動流程：
    1. 讀 ds_yaml/*.yaml（透過 logic.config 的 SOURCE_CONFIGS）
    2. 建立 GStreamer Pipeline（pgie 車 → tracker → analytics）
    3. 為每路 cam 組合下游分支（save / show / rtsp_push）
    4. 若有任何 cam 啟用 rtsp_push，啟動 GstRtspServer 對外提供推流
    5. 進入主迴圈，按 Q 鍵安全退出

RTSP 推流（選項 A：每路獨立）：
    每路 cam 透過 udpsink 把 RTP 封包送到 127.0.0.1:{5400+pad_index}
    再由 GstRtspServer 從這些 loopback 端口讀取，對外提供統一的 RTSP URL：
        rtsp://<邊緣機IP>:<port>/<mount_path>
    多路可共用同一個 port（用 mount_path 區分）。
"""

import sys
import time
import select
import termios
import tty

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer

from logic.color import load_labels, CLASS_MAP
# ⭐ 移除二級推論設定的 import：INFER_SEC_PLATE_CONFIG / INFER_SEC_NUM_CONFIG
from logic.config import (
    SOURCE_CONFIGS, INFER_CONFIG, TRACKER_CONFIG,
    PREPROCESS_CONFIG, ANALYTICS_CONFIG,
)
from logic.state import initialize_state_managers, force_finalize_all
from logic.state_db import initialize_state_managers, force_finalize_all
from logic.pipeline import (
    cb_newpad, cb_source_setup, make_elm,
    _build_display_sink, setup_cam_branch
)
# ⭐ 移除車牌 / 字元相關探針的 import：expand_plate_probe / assemble_plate_probe
from logic.probes import (
    tracker_src_pad_buffer_probe, per_cam_osd_probe,
)

g_loop = None
g_pipeline = None
g_eos_triggered = False
g_rtsp_server = None  # 持有 RTSP server 引用，避免 GC 掉


def force_quit_loop():
    """EOS 超時的強制退出 fallback（避免影片寫不出時卡住）。"""
    global g_loop
    print("\n[WARNING] 等待影片封裝逾時，強制退出管線！")
    if g_loop and g_loop.is_running():
        g_loop.quit()
    return False


def keyboard_cb(fd, condition):
    """終端機按鍵處理，按 Q 觸發 EOS 安全退出。"""
    global g_eos_triggered, g_pipeline, g_loop
    ch = sys.stdin.read(1)
    if ch in ('q', 'Q') and not g_eos_triggered:
        g_eos_triggered = True
        print("\n[INFO] 收到 'Q' 鍵，正在安全發送 EOS 訊號 (等待影片寫入)...")
        if g_pipeline:
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
        return False
    return True


def bus_call(bus, message, loop):
    """GStreamer bus 訊息處理。RTSP 訊號錯誤忽略以利自動重連。"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[INFO] 影像串流結束 (EOS 處理完畢)，準備安全退出...")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        err_msg = str(err).lower()
        if "rtsp" in err_msg or "timeout" in err_msg or "resource not found" in err_msg or "could not read" in err_msg:
            print(f"[WARNING] RTSP 串流不穩或中斷: {err}。系統保持運行，等待自動重連...")
        else:
            print(f"[ERROR] 嚴重管線錯誤: {err}: {debug}")
            loop.quit()
    return True


def _enlarge_queue(q, max_buffers=400):
    """放寬 queue 容量，避免下游處理偶發較慢時被反壓。"""
    q.set_property("max-size-buffers", max_buffers)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)


# ==========================================
# ⭐ RTSP server 啟動：把每路 udpsink 端口註冊成 mount_path
# ==========================================
def _start_rtsp_server(rtsp_routes):
    """
    依照各路 cam 的 RTSP 設定，啟動 GstRtspServer 並建立對應的 mount points。

    參數：
        rtsp_routes (list of dict): 每筆包含
            - pad_index   : 第幾路
            - udp_port    : 該路 udpsink 用的 loopback port（5400+pad_index）
            - port        : RTSP 對外服務 port（通常每路都一樣，例如 8554）
            - mount_path  : URL 路徑（例：camC、camD）
            - encoder     : "h264" 或 "h265"

    回傳：
        GstRtspServer.RTSPServer 物件（呼叫端必須保留引用避免 GC）。

    對外 URL：
        rtsp://<本機IP>:<port>/<mount_path>
    """
    if not rtsp_routes:
        return None

    # 用 port 分組（理論上只會有一個 port，但保險起見支援多 port）
    routes_by_port = {}
    for r in rtsp_routes:
        routes_by_port.setdefault(r["port"], []).append(r)

    # 每個 port 開一台 RTSP server
    servers = []
    for port, routes in routes_by_port.items():
        server = GstRtspServer.RTSPServer()
        server.set_service(str(port))
        mounts = server.get_mount_points()

        for r in routes:
            udp_port = r["udp_port"]
            encoder = r["encoder"]
            mount_path = "/" + r["mount_path"].lstrip("/")  # 確保開頭 /

            # 中央接收端 SDP（描述串流格式給客戶端 VLC/ffplay）
            # 注意：encoding-name 必須跟 udpsink 那端的 rtp{264/5}pay 對應
            if encoder == "h265":
                enc_name = "H265"
            else:
                enc_name = "H264"

            launch_str = (
                f"( udpsrc port={udp_port} caps=\"application/x-rtp, "
                f"media=video, clock-rate=90000, encoding-name={enc_name}, payload=96\" "
                f"! rtp{encoder}depay ! rtp{encoder}pay name=pay0 pt=96 )"
            )

            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(launch_str)
            factory.set_shared(True)   # 多客戶端可同時連同一 mount
            mounts.add_factory(mount_path, factory)

            print(f"[INFO] RTSP 推流註冊: rtsp://<本機IP>:{port}{mount_path}  (encoder={encoder}, udp_port={udp_port})")

        server.attach(None)
        servers.append(server)

    return servers


def main():
    global g_loop, g_pipeline, g_eos_triggered, g_rtsp_server

    # ⭐ 啟動訊息改為單層架構描述
    print("[INFO] 初始化 DeepStream 車流單層架構 (PGIE 車輛偵測 + Tracker)...")

    Gst.init(None)
    g_pipeline = Gst.Pipeline.new("traffic-pipeline")

    num_sources = len(SOURCE_CONFIGS)

    # 任一 cam 開啟 show_window 就建立 display sink
    show_window = any(cfg.get("display", {}).get("show_window", True) for cfg in SOURCE_CONFIGS.values())

    # ⭐ 判斷是否有任何 live 來源（RTSP/HTTP/HTTPS），決定 streammux 時鐘模式
    #     - 全部都是 file://      → live-source=0（依 buffer PTS 跑）
    #     - 任一是 RTSP/HTTP 等  → live-source=1（依 wall clock 跑）
    #   這個值與下游 display streammux 必須一致，否則 pipeline 會卡 PAUSED。
    has_live_source = any(
        not cfg.get("is_file_source", False) for cfg in SOURCE_CONFIGS.values()
    )
    print(f"[INFO] 來源模式: {'live (RTSP/HTTP)' if has_live_source else 'file (本地檔案)'}")

    # === 主 streammux ===
    streammux = make_elm("nvstreammux", "Stream-muxer")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", 70000)
    streammux.set_property("live-source", 1 if has_live_source else 0)
    streammux.set_property("nvbuf-memory-type", 0)
    g_pipeline.add(streammux)

    # === 來源 decodebin ===
    for pad_index, cfg in SOURCE_CONFIGS.items():
        source = make_elm("uridecodebin", f"uri-decode-bin-{pad_index}")
        source.set_property("uri", cfg["source"])
        source.connect("pad-added", cb_newpad, {"streammux": streammux, "pad_index": pad_index})
        source.connect("source-setup", cb_source_setup, None)
        g_pipeline.add(source)

    # === Queue / 推論元件 ===
    # ⭐ 移除 q_sgie_plate / q_sgie_num 兩個 queue（不再有 SGIE）
    q1 = make_elm("queue", "q1")
    q2 = make_elm("queue", "q2")
    q3 = make_elm("queue", "q3")
    q_analytics = make_elm("queue", "q_analytics")
    q4 = make_elm("queue", "q4")

    # ⭐ 同步移除對應的 _enlarge_queue 呼叫（q_sgie_plate / q_sgie_num）
    _enlarge_queue(q_analytics, max_buffers=200)

    preprocess = make_elm("nvdspreprocess", "preprocess")
    preprocess.set_property("config-file", PREPROCESS_CONFIG)

    pgie = make_elm("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", INFER_CONFIG)
    pgie.set_property("input-tensor-meta", True)

    tracker = make_elm("nvtracker", "tracker")
    tracker.set_property("ll-config-file", TRACKER_CONFIG)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)

    # ⭐ 移除：sgie_plate / sgie_num 兩個 nvinfer 元件不再建立

    analytics = make_elm("nvdsanalytics", "analytics")
    analytics.set_property("config-file", ANALYTICS_CONFIG)

    # ⭐ 加入 pipeline 的元件 list 移除 q_sgie_plate / sgie_plate / q_sgie_num / sgie_num
    for elm in [q1, preprocess, q2, pgie, q3, tracker,
                q_analytics, analytics, q4]:
        g_pipeline.add(elm)

    # ⭐ 連結順序改為：tracker → q_analytics → analytics → q4
    #     原本：tracker → q_sgie_plate → sgie_plate → q_sgie_num → sgie_num → q_analytics → analytics → q4
    streammux.link(q1); q1.link(preprocess); preprocess.link(q2)
    q2.link(pgie); pgie.link(q3); q3.link(tracker)
    tracker.link(q_analytics)
    q_analytics.link(analytics); analytics.link(q4)

    # === Probes ===
    # ⭐ 移除：
    #     sgie_plate.src 的 expand_plate_probe
    #     sgie_num.src   的 assemble_plate_probe
    # 僅保留 tracker.src 的車輛狀態更新探針
    tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)

    # === Demux 並組各路下游分支 ===
    demux = make_elm("nvstreamdemux", "demuxer")
    g_pipeline.add(demux)
    q4.link(demux)

    display_streammux = (
        _build_display_sink(g_pipeline, num_sources, has_live_source=has_live_source)
        if show_window else None
    )    
    # ⭐ 收集所有啟用 RTSP 推流的路，等下批次註冊到 RTSP server
    rtsp_routes = []
    for pad_index, cfg in SOURCE_CONFIGS.items():
        udp_port = setup_cam_branch(g_pipeline, pad_index, cfg, demux, display_streammux, per_cam_osd_probe)
        if udp_port is not None:
            rtsp_routes.append({
                "pad_index":  pad_index,
                "udp_port":   udp_port,
                "port":       cfg["rtsp_push"]["port"],
                "mount_path": cfg["rtsp_push"]["mount_path"],
                "encoder":    cfg["rtsp_push"]["encoder"],
            })

    # === 啟動 RTSP server（只在有任何 cam 推流時才啟動）===
    if rtsp_routes:
        g_rtsp_server = _start_rtsp_server(rtsp_routes)
        print(f"[INFO] 共 {len(rtsp_routes)} 條 RTSP 推流就緒")
    else:
        print("[INFO] 無 cam 啟用 RTSP 推流，跳過 RTSP server 啟動")

    # === 鍵盤監聽 + 主迴圈 ===
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, keyboard_cb)
        print("\n[INFO] 💡 提示：在終端機按下 'q' 鍵即可優雅退出並存檔...\n")

        g_loop = GLib.MainLoop()
        bus = g_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, g_loop)

        g_pipeline.set_state(Gst.State.PLAYING)
        g_loop.run()

    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，準備發送 EOS...")
        if not g_eos_triggered:
            g_eos_triggered = True
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
            try:
                g_loop.run()
            except KeyboardInterrupt:
                print("\n[INFO] 強制終止！")
                pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        force_finalize_all()
        g_pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    initialize_state_managers()
    main()