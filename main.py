import cv2, json, os, numpy as np, prepare_stream
from ultralytics import YOLO

SPOTS_FILE = "spots.json"
W, H, SCHEMA_W = 1280, 720, 450
DENSITY_THRESHOLD = 0.18  # CV Modu için piksel yoğunluk eşiği (%18)
VEHICLES = [2, 3, 5, 7]   # YOLO Modu için araç sınıfları (Car, Motor, Bus, Truck)

def load_spots():
    if not os.path.exists(SPOTS_FILE): return []
    with open(SPOTS_FILE, "r", encoding="utf-8") as f: return json.load(f)

def check_occupancy_cv(spot, dilate_img):
    # spot: Park yeri bilgilerini içeren sözlük. "points" anahtarı çokgen köşelerini tutar.
    # dilate_img: Ön işleme sonrası genleşmiş (dilate) gri/binary görüntü.
    spot_pts = np.array(spot["points"], np.int32)  # Polygon noktalarını NumPy dizisine çevir.
    # cv2.boundingRect(): Çokgenin en küçük dikdörtgenini döndürür -> (x, y, w, h).
    x, y, w, h = cv2.boundingRect(spot_pts)
    # Geçersiz veya sıfır boyutlu alan için doğrudan boş kabul et.
    if w <= 0 or h <= 0: return False
    # Mask: Park alanının içindeki piksel bölgelerini temsil eden 2D siyah-beyaz görüntü.
    mask = np.zeros((h, w), dtype=np.uint8)
    # Polygonu kapsayan dikdörtgenin içindeki koordinatlara göre yerleştir ve doldur.
    cv2.fillPoly(mask, [spot_pts - [x, y]], 255)
    # Mask içindeki toplam piksel sayısı = park alanının gerçek alanı.
    total_pixels = cv2.countNonZero(mask)
    # Alan boşsa işlem yapma.
    if total_pixels <= 0: return False
    # Sadece ilgili park alanının dikdörtgen kesitini kırp.
    crop = dilate_img[y:y+h, x:x+w]
    # Kırpılmış görüntü ile park alanı maskesinin kesişimini al.
    intersection = cv2.bitwise_and(crop, mask)
    # Kesişim piksel oranı, eşik değerden büyükse park alanı dolu kabul edilir.
    return (cv2.countNonZero(intersection) / total_pixels) >= DENSITY_THRESHOLD

def check_occupancy_yolo(spot, vehicle_boxes, min_overlap=0.30):
    # spot: Park yeri bilgilerini içeren sözlük. "points" anahtarı çokgen köşelerini tutar.
    # vehicle_boxes: YOLO tarafından tespit edilen araç kutularının [x1, y1, x2, y2] koordinatları listesi.
    # min_overlap: Park alanının minimum kaç yüzdesinin araç tarafından işgal edilmesi gerektiği (varsayılan: %30).
    
    spot_pts = np.array(spot["points"], np.int32)
    # Park alanının en küçük çevreleyen dikdörtgenini al (x, y, genişlik, yükseklik)
    x, y, w, h = cv2.boundingRect(spot_pts)
    # Geçersiz veya sıfır boyutlu alan için doğrudan False döndür
    if w <= 0 or h <= 0: return False

    # Park alanı için maske oluştur (h×w boyutunda siyah görüntü)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Çokgeni mask üzerine doldur (koordinatları çevreleyen dikdörtgene göre kaydır)
    cv2.fillPoly(mask, [spot_pts - [x, y]], 255)
    # Park alanının toplam piksel sayısını say
    total_pixels = cv2.countNonZero(mask)
    # Park alanı boşsa işlem yapma
    if total_pixels <= 0: return False

    # Her bir tespit edilen araç kutusunu kontrol et
    for box in vehicle_boxes:
        # Kutunun köşelerini al (x1, y1 = sol üst; x2, y2 = sağ alt)
        bx1, by1, bx2, by2 = map(int, box)
        # Kutunun merkezini hesapla
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        # Kutu merkezi park alanı içinde mi? (>= 0 ise içinde demektir)
        if cv2.pointPolygonTest(spot_pts, (cx, cy), False) >= 0: return True

        # Kutu ile park alanının kesişim alanını belirle
        ix1, iy1 = max(0, bx1 - x), max(0, by1 - y)
        ix2, iy2 = min(w, bx2 - x), min(h, by2 - y)
        # Kesişim alanı geçerli mi kontrol et
        if ix2 > ix1 and iy2 > iy1:
            # Kesişim piksel oranı minimum eşikten büyükse park alanı dolu
            if (np.count_nonzero(mask[iy1:iy2, ix1:ix2]) / total_pixels) >= min_overlap:
                return True
    # Hiçbir araç park alanını işgal etmiyorsa False döndür
    return False

def render_schema(spots_status, mode):
    # Panel oluştur ve arka plan rengi ayarla
    panel = np.zeros((H, SCHEMA_W, 3), dtype=np.uint8)
    panel[:] = (24, 28, 36)

    # Başlık bölümü
    cv2.rectangle(panel, (0, 0), (SCHEMA_W, 105), (38, 42, 54), -1)
    cv2.putText(panel, "CANLI OTOPARK SEMASI", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # İstatistikleri hesapla
    total = len(spots_status)
    occupied = sum(1 for s in spots_status if s["occupied"])
    empty = total - occupied

    # İstatistik metinlerini göster
    cv2.putText(panel, f"Toplam: {total}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(panel, f"BOS: {empty}", (130, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 2)
    cv2.putText(panel, f"DOLU: {occupied}", (240, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 2)
    cv2.putText(panel, f"AKTIF MOD: {mode} (Degistir: 'm')", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.line(panel, (0, 105), (SCHEMA_W, 105), (60, 65, 80), 2)

    # Her park alanını dikdörtgen olarak çiz
    for s in spots_status:
        bx, by, bw, bh = s["sx"], s["sy"], s["sw"], s["sh"]
        is_occ = s["occupied"]
        # Doluluk durumuna göre renklendir (kırmızı/yeşil)
        cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (40, 40, 175) if is_occ else (35, 145, 55), -1)
        cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (80, 80, 255) if is_occ else (80, 255, 120), 2)
        cv2.putText(panel, f"P{s['id']}", (bx + 6, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(panel, "DOLU" if is_occ else "BOS", (bx + 6, by + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return panel

def main():
    spots = load_spots()
    if not spots: print("[UYARI] Tanımlı park alanı yok! Önce 'python editor.py' çalıştırın.")

    print("[INFO] Yapay Zeka Modeli (YOLOv8) Yükleniyor...")
    model = YOLO("yolov8n.pt")

    cap = prepare_stream.get_stream()
    if not cap.isOpened(): return

    win = "Otopark Canli Izleme Modu"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W + SCHEMA_W, H)

    mode, show_threshold = "YOLO", False
    frame_count, vehicle_boxes = 0, []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        frame = cv2.resize(frame, (W, H))
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        median = cv2.medianBlur(thresh, 5)
        dilate = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

        # YOLO modunda her 2 frame'de bir veya başlangıçta YOLO modelini çalıştır ve araç kutularını güncelle
        if mode == "YOLO" and (frame_count % 2 == 0 or not vehicle_boxes):
            results = model(frame, verbose=False, conf=0.15)[0]
            # Tespit edilen nesnelerden tanımlı araç sınıflarını (araba, motor, otobüs, kamyon) filtrele
            vehicle_boxes = [box.xyxy[0].tolist() for box in results.boxes if int(box.cls[0]) in VEHICLES]

        # Tüm park alanlarının doluluk durumunu kontrol et ve saklı sözlüğü oluştur
        spots_status = []
        for s in spots:
            # Seçilen moda göre park alanının dolu olup olmadığını belirle
            occupied = check_occupancy_yolo(s, vehicle_boxes) if mode == "YOLO" else check_occupancy_cv(s, dilate)
            # Park alanı bilgilerine "occupied" anahtarını ekle ve listeye ekle
            spots_status.append({**s, "occupied": occupied})

            pts = np.array(s["points"], np.int32).reshape((-1, 1, 2))
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(frame, [pts], True, color, 2)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            M = cv2.moments(pts)
            if M["m00"]:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(frame, f"P{s['id']}:{'DOLU' if occupied else 'BOS'}", (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if mode == "YOLO":
            for box in vehicle_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 50), 2)

        display_frame = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR) if (show_threshold and mode == "CV") else frame
        cv2.imshow(win, np.hstack((display_frame, render_schema(spots_status, mode))))

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27: break
        elif key == ord('r'): spots = load_spots()
        elif key == ord('t'): show_threshold = not show_threshold
        elif key == ord('m'): mode = "YOLO" if mode == "CV" else "CV"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
