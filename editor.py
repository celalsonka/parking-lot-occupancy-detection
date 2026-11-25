import cv2, json, os, numpy as np, prepare_stream

SPOTS_FILE = "spots.json"
W, H, SCHEMA_W = 1280, 720, 450
WIN_NAME = "OTOPARK EDIT MODU"

spots, active_pts, sel_idx, drag_mode, ox, oy = [], [], None, None, 0, 0
snap_enabled, snap_tol = True, 5
active_v_lines, active_h_lines = [], []
default_w, default_h = 85, 50

def load_spots():
    global spots, default_w, default_h
    if os.path.exists(SPOTS_FILE):
        with open(SPOTS_FILE, "r", encoding="utf-8") as f:
            spots = json.load(f)
            if spots: default_w, default_h = spots[-1]["sw"], spots[-1]["sh"]

def save_spots():
    with open(SPOTS_FILE, "w", encoding="utf-8") as f: json.dump(spots, f, indent=2)
    print(f"[KAYDEDİLDİ] {len(spots)} park alanı kaydedildi.")

def mouse_handler(event, x, y, flags, param):
    global active_pts, spots, sel_idx, drag_mode, ox, oy, active_v_lines, active_h_lines, default_w, default_h

    # 1. KAMERA EKRANI (Sol Panel)
    if x < W:
        if event == cv2.EVENT_LBUTTONDOWN:
            active_pts.append([x, y])
            if len(active_pts) == 4:
                sid = len(spots) + 1
                spots.append({
                    "id": sid, "points": active_pts,
                    "sx": 15 + ((sid - 1) % 4) * 105, "sy": 105 + ((sid - 1) // 4) * 65,
                    "sw": default_w, "sh": default_h
                })
                active_pts = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            if active_pts: active_pts = []
            else:
                for i, s in enumerate(spots):
                    if cv2.pointPolygonTest(np.array(s["points"], np.int32), (x, y), False) >= 0:
                        spots.pop(i)
                        for idx, item in enumerate(spots): item["id"] = idx + 1
                        break
        return

    # 2. ŞEMA EKRANI (Sağ Panel)
    sx, sy = x - W, y
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in reversed(range(len(spots))):
            s = spots[i]
            bx, by, bw, bh = s["sx"], s["sy"], s["sw"], s["sh"]
            if (bx + bw - 14 <= sx <= bx + bw + 14) and (by + bh - 14 <= sy <= by + bh + 14):
                sel_idx, drag_mode = i, 'resize'; break
            elif (bx <= sx <= bx + bw) and (by <= sy <= by + bh):
                sel_idx, drag_mode, ox, oy = i, 'move', sx - bx, sy - by; break

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, s in enumerate(spots):
            if (s["sx"] <= sx <= s["sx"] + s["sw"]) and (s["sy"] <= sy <= s["sy"] + s["sh"]):
                spots.pop(i)
                for idx, item in enumerate(spots): item["id"] = idx + 1
                break

    elif event == cv2.EVENT_MOUSEMOVE and sel_idx is not None:
        s = spots[sel_idx]
        if drag_mode == 'move':
            target_sx, target_sy = sx - ox, sy - oy
            active_v_lines, active_h_lines = [], []

            if snap_enabled:
                for o in spots:
                    if o["id"] == s["id"]: continue
                    # 5px Dikey / Sütun Kilitleme (sx)
                    if abs(target_sx - o["sx"]) <= snap_tol:
                        target_sx = o["sx"]; active_v_lines.append(o["sx"])
                    elif abs((target_sx + s["sw"]) - (o["sx"] + o["sw"])) <= snap_tol:
                        target_sx = o["sx"] + o["sw"] - s["sw"]; active_v_lines.append(o["sx"] + o["sw"])
                    elif abs((target_sx + s["sw"] / 2) - (o["sx"] + o["sw"] / 2)) <= snap_tol:
                        target_sx = o["sx"] + (o["sw"] - s["sw"]) // 2; active_v_lines.append(int(o["sx"] + o["sw"] / 2))

                    # 5px Yatay / Satır Kilitleme (sy)
                    if abs(target_sy - o["sy"]) <= snap_tol:
                        target_sy = o["sy"]; active_h_lines.append(o["sy"])
                    elif abs((target_sy + s["sh"]) - (o["sy"] + o["sh"])) <= snap_tol:
                        target_sy = o["sy"] + o["sh"] - s["sh"]; active_h_lines.append(o["sy"] + o["sh"])
                    elif abs((target_sy + s["sh"] / 2) - (o["sy"] + o["sh"] / 2)) <= snap_tol:
                        target_sy = o["sy"] + (o["sh"] - s["sh"]) // 2; active_h_lines.append(int(o["sy"] + o["sh"] / 2))
                    # 5px Dikey Boşluk Hizalama
                    elif abs((target_sy + s["sh"]) - (o["sy"] - 5)) <= snap_tol:
                        target_sy = o["sy"] - s["sh"] - 5; active_h_lines.append(o["sy"] - 5)
                    elif abs(target_sy - (o["sy"] + o["sh"] + 5)) <= snap_tol:
                        target_sy = o["sy"] + o["sh"] + 5; active_h_lines.append(o["sy"] + o["sh"] + 5)

            s["sx"] = max(5, min(target_sx, SCHEMA_W - s["sw"] - 5))
            s["sy"] = max(55, min(target_sy, H - s["sh"] - 5))

        elif drag_mode == 'resize':
            new_w = max(35, min(sx - s["sx"], SCHEMA_W - s["sx"] - 5))
            new_h = max(25, min(sy - s["sy"], H - s["sy"] - 5))
            default_w, default_h = new_w, new_h
            for spot in spots: spot["sw"], spot["sh"] = new_w, new_h

    elif event == cv2.EVENT_LBUTTONUP:
        sel_idx, drag_mode = None, None
        active_v_lines, active_h_lines = [], []

def main():
    global active_pts, spots, snap_enabled, snap_tol
    load_spots()
    cap = prepare_stream.get_stream()
    ret, frame = cap.read()
    cap.release()
    if not ret: return

    frame = cv2.resize(frame, (W, H))
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, W + SCHEMA_W, H)
    cv2.setMouseCallback(WIN_NAME, mouse_handler)

    while True:
        cam_view = frame.copy()
        for s in spots:
            pts = np.array(s["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(cam_view, [pts], True, (0, 255, 0), 2)
            M = cv2.moments(pts)
            if M["m00"]: cv2.putText(cam_view, f"P{s['id']}", (int(M["m10"]/M["m00"]) - 12, int(M["m01"]/M["m00"]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for pt in active_pts: cv2.circle(cam_view, tuple(pt), 4, (0, 0, 255), -1)
        if len(active_pts) > 1:
            for i in range(len(active_pts) - 1): cv2.line(cam_view, tuple(active_pts[i]), tuple(active_pts[i+1]), (0, 255, 255), 2)
            if len(active_pts) == 4: cv2.line(cam_view, tuple(active_pts[3]), tuple(active_pts[0]), (0, 255, 255), 2)

        schema_view = np.zeros((H, SCHEMA_W, 3), dtype=np.uint8)
        schema_view[:] = (24, 28, 36)
        cv2.rectangle(schema_view, (0, 0), (SCHEMA_W, 45), (38, 42, 54), -1)
        cv2.putText(schema_view, "SEMA DUZLEMI (X,Y)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(schema_view, (0, 45), (SCHEMA_W, 45), (60, 65, 80), 2)

        # Kılavuz Hizalama Çizgilerini Çiz
        for vx in active_v_lines: cv2.line(schema_view, (vx, 45), (vx, H - 5), (255, 0, 255), 1)
        for vy in active_h_lines: cv2.line(schema_view, (5, vy), (SCHEMA_W - 5, vy), (255, 0, 255), 1)

        for i, s in enumerate(spots):
            bx, by, bw, bh = s["sx"], s["sy"], s["sw"], s["sh"]
            cv2.rectangle(schema_view, (bx, by), (bx + bw, by + bh), (45, 55, 75) if i == sel_idx else (35, 45, 60), -1)
            cv2.rectangle(schema_view, (bx, by), (bx + bw, by + bh), (0, 255, 255) if i == sel_idx else (80, 255, 120), 2)
            cv2.putText(schema_view, f"P{s['id']}", (bx + 8, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.rectangle(schema_view, (bx + bw - 8, by + bh - 8), (bx + bw, by + bh), (0, 255, 255), -1)

        combined = np.hstack((cam_view, schema_view))
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 42), (20, 24, 32), -1)
        cv2.putText(combined, "HAMLELER: [Kamera Sol-Tik: Alan Ekle] [Sema Sol-Tik: Tasi/Resize] [Sag-Tik: Sil] | TUSLAR: 's':Kaydet 'd':SonSil 'r':Sifirla 'a':Hizala 'q':Cikis", 
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 255, 255), 1)
        cv2.line(combined, (0, 42), (combined.shape[1], 42), (60, 65, 80), 2)

        cv2.imshow(WIN_NAME, combined)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'): save_spots()
        elif key == ord('d') and spots: spots.pop(); active_pts = []
        elif key == ord('r'): spots, active_pts = [], []
        elif key == ord('a'): snap_enabled = not snap_enabled
        elif key in [ord('+'), ord('=')]: snap_tol = min(30, snap_tol + 2)
        elif key == ord('-'): snap_tol = max(2, snap_tol - 2)
        elif key == ord('q') or key == 27: save_spots(); break

    cv2.destroyAllWindows()

if __name__ == "__main__": main()
