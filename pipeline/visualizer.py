import cv2
import numpy as np


# ── Warna utama ───────────────────────────────────────────────────────────────
COLOR_VEHICLE = (57, 255, 20)   # hijau neon — pojok sudut bbox kendaraan
COLOR_PLATE   = (255, 50, 50)   # merah — bbox plat
COLOR_TEXT_BG = (255, 255, 255) # putih — background teks hasil baca
COLOR_TEXT_FG = (20, 20, 20)    # hitam pekat — teks

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ── Helper: gambar sudut-sudut saja (bukan garis penuh) ───────────────────────

def _draw_corner_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     color: tuple, thickness: int = 3, ratio: float = 0.18) -> None:
    """Gambar 4 sudut bbox ala referensi gambar (bukan kotak penuh)."""
    w, h = x2 - x1, y2 - y1
    lx = max(12, int(w * ratio))   # panjang garis horizontal tiap sudut
    ly = max(12, int(h * ratio))   # panjang garis vertikal tiap sudut

    corners = [
        # (titik pojok, (dx_h, 0), (0, dy_v))  — arah garis keluar dari pojok
        ((x1, y1), (+lx, 0), (0, +ly)),  # kiri-atas
        ((x2, y1), (-lx, 0), (0, +ly)),  # kanan-atas
        ((x1, y2), (+lx, 0), (0, -ly)),  # kiri-bawah
        ((x2, y2), (-lx, 0), (0, -ly)),  # kanan-bawah
    ]
    for (cx, cy), (dx, _), (_, dy) in corners:
        cv2.line(frame, (cx, cy), (cx + dx, cy),        color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx,      cy + dy),   color, thickness, cv2.LINE_AA)


# ── Fungsi utama yang dipanggil dari app.py ───────────────────────────────────

def draw_vehicle_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    plate_text: str | None = None,
    plate_bbox_abs: tuple | None = None,   # (px1,py1,px2,py2) absolut di frame
    plate_crop: np.ndarray | None = None,
) -> None:
    """
    Gambar semua overlay untuk satu kendaraan:
      1. Sudut-sudut bbox kendaraan (hijau neon)
      2. ID kendaraan kecil di pojok kiri-atas bbox
      3. Bbox plat merah (jika plate_bbox_abs ada)
      4. Crop plat di dalam bbox kendaraan (pojok kanan-bawah bbox kendaraan)
      5. Teks hasil baca tepat di atas crop plat (background putih, lebar = crop plat)
    """
    # 1. Corner bbox kendaraan ─────────────────────────────────────────────────
    _draw_corner_box(frame, x1, y1, x2, y2, COLOR_VEHICLE, thickness=3)

    # 2. ID track kecil ────────────────────────────────────────────────────────
    id_label = f"ID {track_id}"
    (iw, ih), _ = cv2.getTextSize(id_label, FONT, 0.45, 1)
    # tempel di pojok kiri-atas, tepat di dalam sudut
    ix, iy = x1 + 6, y1 + ih + 6
    cv2.putText(frame, id_label, (ix, iy), FONT, 0.45, COLOR_VEHICLE, 1, cv2.LINE_AA)

    # 3. Bbox plat merah ───────────────────────────────────────────────────────
    if plate_bbox_abs is not None:
        px1, py1, px2, py2 = plate_bbox_abs
        cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_PLATE, 2, cv2.LINE_AA)

    # 4 + 5. Crop plat + label teks, di dalam bbox mobil ──────────────────────
    if plate_crop is not None and plate_crop.size > 0:
        _embed_plate_inside_vehicle(
            frame, plate_crop, plate_text,
            vx1=x1, vy1=y1, vx2=x2, vy2=y2
        )


def _embed_plate_inside_vehicle(
    frame: np.ndarray,
    plate_crop: np.ndarray,
    plate_text: str | None,
    vx1: int, vy1: int, vx2: int, vy2: int,
    target_h: int = 36,
    padding: int = 8,
) -> None:
    """
    Pasang teks hasil baca + crop plat di tengah-atas bbox kendaraan.
    Layout (dari atas ke bawah, center horizontal):
        [ teks hasil baca (background putih) ]
        [ crop plat asli                     ]
    """
    # ── resize crop plat ke tinggi target ────────────────────────────────────
    ph, pw = plate_crop.shape[:2]
    if ph == 0 or pw == 0:
        return
    scale  = target_h / ph
    crop_w = max(1, int(pw * scale))
    crop_h = target_h

    thumb = cv2.resize(plate_crop, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    if len(thumb.shape) == 2:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)

    # ── bangun banner teks (lebar = lebar crop plat) ─────────────────────────
    banner_h = 0
    banner   = None
    if plate_text:
        font_scale = 0.60
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(plate_text, FONT_BOLD, font_scale, thickness)
        banner_h = th + baseline + 6
        banner_w = max(crop_w, tw + 12)   # minimal selebar teks
        crop_w   = banner_w               # selaraskan crop juga
        # re-resize crop agar lebarnya match banner
        thumb = cv2.resize(plate_crop, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
        if len(thumb.shape) == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        banner = np.full((banner_h, banner_w, 3), 255, dtype=np.uint8)
        tx = max(0, (banner_w - tw) // 2)
        ty = th + 2
        cv2.putText(banner, plate_text, (tx, ty), FONT_BOLD,
                    font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA)

    total_h = banner_h + crop_h

    # ── posisi: tengah-atas bbox kendaraan ───────────────────────────────────
    fh, fw  = frame.shape[:2]
    cx      = (vx1 + vx2) // 2
    dest_x1 = max(0, cx - crop_w // 2)
    dest_x2 = min(fw, dest_x1 + crop_w)
    dest_y1 = max(0, vy1 + padding)
    dest_y2 = min(fh, dest_y1 + total_h)

    # ── tempel banner teks (atas) ─────────────────────────────────────────────
    cur_y = dest_y1
    if banner is not None:
        by2 = min(cur_y + banner_h, dest_y2, fh)
        bx2 = min(dest_x1 + crop_w, fw)
        bh  = by2 - cur_y
        bw  = bx2 - dest_x1
        if bh > 0 and bw > 0:
            frame[cur_y:by2, dest_x1:bx2] = banner[:bh, :bw]
        cur_y = by2

    # ── tempel crop plat (bawah) ──────────────────────────────────────────────
    py2 = min(cur_y + crop_h, dest_y2, fh)
    px2 = min(dest_x1 + crop_w, fw)
    ch  = py2 - cur_y
    cw  = px2 - dest_x1
    if ch > 0 and cw > 0:
        frame[cur_y:py2, dest_x1:px2] = thumb[:ch, :cw]
        cv2.rectangle(frame, (dest_x1, cur_y), (px2 - 1, py2 - 1), COLOR_PLATE, 1)