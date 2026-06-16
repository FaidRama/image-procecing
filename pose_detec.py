import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class Human_Awareness_System:
    def __init__(self, model_path="yolov8n-pose.pt"):
        """
        Inisialisasi arsitektur sistem vision dan manajemen memori.
        """
        # Load Model
        self.model = YOLO(model_path)
        
        # Buffer Memori Temporal (Dictionary of Deques)
        self.posisi_history = {}
        
        # SOLUSI MEMORY LEAK: Tracker untuk mencatat seberapa lama ID tidak terlihat
        self.missing_count = {}
        
        # Hyperparameters Sistem
        self.treshold_gerakan = 30       # Jarak piksel gerakan berjalan
        self.kp_conf_treshold = 0.5      # Batas kepercayaan titik koordinat tubuh
        self.max_missing_frames = 45     # Batas toleransi objek hilang (~1.5 detik pada 30 FPS)

    def clean_stale_tracks(self, active_ids):
        """
        Fungsi Manajemen Memori (Garbage Collection) untuk mencegah Memory Leak.
        Menghapus ID yang sudah keluar dari jangkauan kamera melewati batas toleransi.
        """
        # Cari ID di memori RAM yang sudah tidak aktif di frame saat ini
        stale_ids = set(self.posisi_history.keys()) - active_ids
        
        for tid in stale_ids:
            # Naikkan hitungan frame absen untuk ID ini
            self.missing_count[tid] = self.missing_count.get(tid, 0) + 1
            
            # Jika absennya melebihi batas (Grace Period Habis), lakukan POP dari RAM
            if self.missing_count[tid] > self.max_missing_frames:
                self.posisi_history.pop(tid)
                self.missing_count.pop(tid)
                print(f"[Memory Cleanup] ID {tid} resmi di-POP dari RAM karena keluar area.")
                
        # Jika ID yang sempat hilang muncul kembali sebelum batas waktu, reset counter absennya
        for tid in active_ids:
            if tid in self.missing_count:
                self.missing_count[tid] = 0

    def analyze_temporal_behavior(self, track_id, center_x, center_y):
        """
        Mengolah data riwayat koordinat untuk menentukan status Diam vs Berjalan.
        """
        if track_id not in self.posisi_history:
            self.posisi_history[track_id] = deque(maxlen=30)
            
        self.posisi_history[track_id].append((center_x, center_y))
        user_history = self.posisi_history[track_id]
        
        # Default state
        perilaku = f"Kalibrasi ID:{track_id} ({len(user_history)}/30)"
        warna_perilaku = (0, 165, 255) # Oranye
        
        if len(user_history) >= 30:
            posisi_awal = user_history[0]
            posisi_akhir = user_history[-1]
            
            jarak_pergeseran = np.sqrt((posisi_akhir[0] - posisi_awal[0]) ** 2 + 
                                       (posisi_akhir[1] - posisi_awal[1]) ** 2)
            
            if jarak_pergeseran > self.treshold_gerakan:
                perilaku = "Berjalan / Lewat"
                warna_perilaku = (255, 0, 0)  # Biru
            else:
                perilaku = "Diam"
                warna_perilaku = (0, 255, 0)  # Hijau
                
        return perilaku, warna_perilaku

    def process_frame(self, frame):
        """
        Pipeline utama pemrosesan gambar, inferensi AI, dan rendering UI.
        """
        # Jalankan tracking model
        results = self.model.track(frame, persist=True, conf=0.6, verbose=False)
        
        # Set untuk mencatat ID siapa saja yang aktif di frame ini
        active_ids = set()

        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints

            for i, box in enumerate(boxes):
                # Ekstrak ID pelacakan objek
                track_id = int(box.id[0]) if box.id is not None else i
                active_ids.add(track_id)

                # Inisialisasi variabel default sistem pengaman
                warna_box = (0, 255, 0)          
                state_jarak = "Menghitung..."
                target_width = 0
                metode_deteksi = "None"

                # Ekstrak koordinat kotak luar objek
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int(x1 + x2) // 2
                center_y = int(y1 + y2) // 2

                # 1. Panggil Analisis Perilaku Temporal (Berjalan vs Diam)
                perilaku, warna_text = self.analyze_temporal_behavior(track_id, center_x, center_y)

                # --- LOGIKA 3: ESTIMASI JARAK HIERARKIS & POSE SPASIAL ---
                if keypoints is not None and len(keypoints.xy) > i:
                    kp = keypoints.xy[i].cpu().numpy()
                    kp_conf = keypoints.conf[i].cpu().numpy()
    
                    # 1. BEDAKAN THRESHOLD (Bahu harus lebih strict biar gampang fallback)
                    CONF_BAHU = 0.5    # Naikkan ke 50%
                    CONF_PINGGUL = 0.4 # Pinggul 40%
                    CONF_WRIST = 0.3   # Tangan lebih rendah gapapa, tangkapan gerak cepat soalnya
    
                    shoulder_visible = kp_conf[5] > CONF_BAHU and kp_conf[6] > CONF_BAHU
                    hips_visible = kp_conf[11] > CONF_PINGGUL and kp_conf[12] > CONF_PINGGUL
                    wrist_visible = kp_conf[9] > CONF_WRIST and kp_conf[10] > CONF_WRIST
    
                    # 2. ANATOMICAL SANITY CHECK (Detektor AI Kesurupan)
                    if shoulder_visible and hips_visible:
                        # Cek apakah Pinggul (Y) berada di atas Bahu (Y)? 
                        # Atau jarak Bahu dan Pinggul kurang dari 30 piksel (berhimpitan/gepeng)?
                        jarak_y_tubuh = kp[11][1] - kp[5][1] # Y_PinggulKiri - Y_BahuKiri
                        
                        if jarak_y_tubuh < 30:
                            # Logika: Mustahil pinggul ada di leher. Berarti model halusinasi (objek terlalu dekat)!
                            # Kita paksa bahu jadi tidak valid agar otomatis fallback ke pinggul/blind
                            shoulder_visible = False
    
                    # Penentuan Jarak Berdasarkan Ketersediaan Fitur Tubuh
                    if shoulder_visible:
                        target_width = np.sqrt((kp[6][0] - kp[5][0]) ** 2 + (kp[6][1] - kp[5][1]) ** 2)
                        metode_deteksi = "Bahu (Stabil)"
                        warna_fitur = (255, 0, 255)  # Magenta
                        cv2.circle(frame, tuple(map(int, kp[5])), 6, warna_fitur, -1)
                        cv2.circle(frame, tuple(map(int, kp[6])), 6, warna_fitur, -1)
                        # Otomatis masuk sini kalau bahu terpotong layar atau dianggap "gila" oleh Sanity Check
                        if hips_visible:
                            # Cari titik tengah bahu dan titik tengah pinggul
                            y_bahu_avg = (kp[5][1] + kp[6][1]) / 2
                            y_pinggul_avg = (kp[11][1] + kp[12][1]) / 2
                            tinggi_torso = y_pinggul_avg - y_bahu_avg
                            
                            # Cek Rasio: Jika lebar bahu < 40% dari tinggi torso, pasti dia nyamping!
                            if target_width < (tinggi_torso * 0.4):
                                # Kalibrasi ulang target_width pakai tinggi torso (dikalikan faktor skala 0.6)
                                # Agar setara dengan piksel bahu pada jarak yang sama
                                target_width = tinggi_torso * 0.6 
                                metode_deteksi = "Torso (Menyamping)"
                                warna_fitur = (0, 165, 255) # Oranye/Amber
                                
                                # Gambar garis lurus "tulang punggung" biar visualisasinya canggih
                                x_bahu_avg = int((kp[5][0] + kp[6][0]) / 2)
                                x_pinggul_avg = int((kp[11][0] + kp[12][0]) / 2)
                                cv2.line(frame, (x_bahu_avg, int(y_bahu_avg)), 
                                                (x_pinggul_avg, int(y_pinggul_avg)), warna_fitur, 4)
    
                    elif hips_visible:
                        # Fallback murni Pinggul (Jika bahu terpotong layar/hilang)
                        target_width = np.sqrt((kp[12][0] - kp[11][0]) ** 2 + (kp[12][1] - kp[11][1]) ** 2)
                        metode_deteksi = "Pinggul (Fallback)"
                        warna_fitur = (255, 255, 0)  # Cyan
                        cv2.circle(frame, tuple(map(int, kp[11])), 6, warna_fitur, -1)
                        cv2.circle(frame, tuple(map(int, kp[12])), 6, warna_fitur, -1)
                    else:
                        target_width = 999  # Nilai ekstrem
                        metode_deteksi = "Blind (Terlalu Dekat)"
                        warna_fitur = (128, 128, 128)  # Abu-abu
                        # Finite State Machine (FSM) Penentu Batas Aman Navigasi
                    if target_width > 150:
                        state_jarak = "Safe Stop (Bahaya)"
                        warna_box = (0, 0, 255)  # Merah
                    elif target_width > 85:
                        state_jarak = "Reduced Speed (Waspada)"
                        warna_box = (0, 165, 255)  # Oranye
                    else:
                        state_jarak = "Safe (Aman)"

                    # Heuristik Deteksi Perilaku Jahil / Agresif (Tangan di atas Bahu)
                    if len(kp) > 10 and kp[5][1] != 0 and kp[9][1] != 0:
                        if wrist_visible and (kp[9][1] < kp[5][1] or kp[10][1] < kp[6][1]):
                            perilaku = "JAHIL / AGRESIF (Tangan Diangkat!)"
                            warna_text = (0, 0, 255)  # Paksa teks berwarna merah

                # 3. Render User Interface (UI) HUD
                cv2.rectangle(frame, (x1, y1), (x2, y2), warna_box, 2)
                cv2.circle(frame, (center_x, center_y), 4, warna_box, -1)
                
                # Render struktur skeleton sendi agar terlihat canggih saat di-CCTV
                if kp is not None:
                    for idx in [5, 6, 9, 10, 11, 12]:
                        if len(kp) > idx and kp[idx][0] != 0:
                            cv2.circle(frame, tuple(map(int, kp[idx])), 4, (255, 255, 255), -1)

                label_jarak = f"Sistem: {state_jarak} | Fitur: {metode_deteksi} ({int(target_width)} px)"
                label_perilaku = f"Aktivitas: {perilaku} [ID: {track_id}]"
                
                cv2.putText(frame, label_jarak, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_box, 2)
                cv2.putText(frame, label_perilaku, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_text, 2)

        # 4. Bersihkan Memori RAM untuk ID yang sudah tidak aktif keluar dari frame
        self.clean_stale_tracks(active_ids)
        return frame

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Instansiasi Class Arsitektur Utama
    analyzer = Human_Awareness_System()
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Proyek CV - Jarak & Perilaku", cv2.WINDOW_GUI_NORMAL)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Kirim frame mentah ke dalam objek class arsitektur
        processed_frame = analyzer.process_frame(frame)
        
        cv2.imshow("Proyek CV - Jarak & Perilaku", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()