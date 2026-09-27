import streamlit as st
import cv2
import easyocr
import numpy as np
import re

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ==================================================
# MODEL PATHS (custom pretrained YOLO11 models)
# ==================================================

CAR_WEIGHTS = "./models/best.pt"
PLATE_WEIGHTS = "./models/best2.pt"

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models(
    car_weights=CAR_WEIGHTS,
    plate_weights=PLATE_WEIGHTS
):
    car_model = YOLO(car_weights)
    plate_model = YOLO(plate_weights)

    return car_model, plate_model


@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])


# ==================================================
# OCR CHARACTER MAPPING
# ==================================================

dict_char_to_int = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'A': '4',
    'G': '6',
    'S': '5',
    'B': '8'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '8': 'B'
}

# ==================================================
# INDONESIAN PLATE REGEX
# ==================================================

PLATE_PATTERN = re.compile(
    r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$'
)

# ==================================================
# CLEAN OCR TEXT
# ==================================================

def clean_plate_text(text):

    text = text.upper()

    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text
    )

    return text


# ==================================================
# FORMAT INDONESIAN PLATE
# ==================================================

def format_indonesian_plate(text):

    text = clean_plate_text(text)

    if not text:
        return None

    first_digit = re.search(r'\d', text)

    if not first_digit:
        return None

    idx = first_digit.start()

    prefix = text[:idx]

    if len(prefix) < 1 or len(prefix) > 2:
        return None

    # =====================================
    # MAXIMUM 4 DIGITS
    # =====================================

    number = ""

    i = idx

    while (
        i < len(text)
        and text[i].isdigit()
        and len(number) < 4
    ):
        number += text[i]
        i += 1

    suffix = text[i:]

    # =====================================
    # OCR CORRECTION
    # =====================================

    prefix = ''.join(
        dict_int_to_char.get(c, c)
        for c in prefix
    )

    number = ''.join(
        dict_char_to_int.get(c, c)
        for c in number
    )

    suffix = ''.join(
        dict_int_to_char.get(c, c)
        for c in suffix
    )

    # =====================================
    # VALIDATION
    # =====================================

    if len(number) < 1 or len(number) > 4:
        return None

    if len(suffix) > 3:
        return None

    candidate = (
        prefix +
        number +
        suffix
    )

    if not PLATE_PATTERN.match(candidate):
        return None

    # =====================================
    # FORMAT OUTPUT
    # =====================================

    if suffix:

        return (
            f"{prefix} "
            f"{number} "
            f"{suffix}"
        )

    return (
        f"{prefix} "
        f"{number}"
    )


# ==================================================
# VISUALIZATION HELPERS
# ==================================================
# Semua elemen ditempel langsung ke gambar asli:
#   - Bbox sudut (corner bracket) hijau untuk deteksi mobil
#   - Bbox persegi merah untuk deteksi plat
#   - Callout putih (teks hitam hasil OCR + crop plat asli)
#     mengambang di atas mobil, terhubung via garis penunjuk

FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "arialbd.ttf",
    "Arial Bold.ttf",
]


def get_bold_font(size):

    # coba beberapa path font TrueType bold yang umum tersedia
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue

    # PENTING: kalau semua kandidat di atas gagal ditemukan,
    # sebelumnya kode jatuh ke ImageFont.load_default() yang
    # ukurannya TETAP KECIL dan MENGABAIKAN parameter size --
    # ini penyebab font tidak berubah walau font_size dinaikkan.
    # Pillow >=9.2 mendukung load_default(size=...) yang beneran
    # bisa di-resize, jadi kita pakai itu sebagai fallback terakhir.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_corner_brackets(
    draw,
    box,
    color=(0, 255, 0),
    length_ratio=0.18,
    thickness=6
):

    x1, y1, x2, y2 = box

    length = int(
        min(x2 - x1, y2 - y1) * length_ratio
    )
    length = max(length, 15)

    corners = [
        ((x1, y1), (1, 0), (0, 1)),   # top-left
        ((x2, y1), (-1, 0), (0, 1)),  # top-right
        ((x1, y2), (1, 0), (0, -1)),  # bottom-left
        ((x2, y2), (-1, 0), (0, -1)),  # bottom-right
    ]

    for (cx, cy), (dxh, dyh), (dxv, dyv) in corners:

        draw.line(
            [(cx, cy), (cx + dxh * length, cy + dyh * length)],
            fill=color,
            width=thickness
        )

        draw.line(
            [(cx, cy), (cx + dxv * length, cy + dyv * length)],
            fill=color,
            width=thickness
        )


def draw_plate_box(draw, box, color=(255, 0, 0), thickness=5):

    x1, y1, x2, y2 = box

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=thickness
    )


def draw_callout(
    canvas,
    anchor_point,
    plate_crop,
    result_text,
    image_size,
    plate_img_width=260,
    font_size=32  # <-- UBAH DI SINI kalau mau font hasil OCR lebih besar/kecil
):
    """
    Menempelkan kotak putih (teks + crop plat) mengambang di atas
    anchor_point (biasanya titik tengah atas bbox plat), lengkap
    dengan garis penunjuk ke arah plat.
    """

    img_w, img_h = image_size

    draw = ImageDraw.Draw(canvas)

    display_text = result_text if result_text else "TIDAK TERBACA"

    font = get_bold_font(font_size)

    # ----- siapkan crop plat -----

    if plate_crop is not None and plate_crop.size > 0:

        plate_pil = Image.fromarray(plate_crop).convert("RGB")

        pw, ph = plate_pil.size

        new_h = max(1, int(ph * (plate_img_width / pw)))

        plate_pil = plate_pil.resize((plate_img_width, new_h))

    else:

        plate_pil = None

    # ----- ukuran teks -----

    text_bbox = draw.textbbox((0, 0), display_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    padding = 24  # jarak teks/gambar plat ke tepi kotak putih
    gap = 14       # jarak antara teks dan crop plat

    box_w = max(text_w, plate_pil.width if plate_pil else 0) + padding * 2

    box_h = padding * 2 + text_h
    if plate_pil:
        box_h += gap + plate_pil.height

    tail_h = 18

    # ----- posisi callout (default di atas anchor, tengah horizontal) -----

    box_x = int(anchor_point[0] - box_w / 2)
    box_y = int(anchor_point[1] - tail_h - box_h)

    place_below = False
    if box_y < 0:
        place_below = True
        box_y = int(anchor_point[1] + tail_h)

    box_x = max(4, min(box_x, img_w - box_w - 4))

    # ----- garis / segitiga penunjuk -----

    tail_top_y = box_y + box_h if not place_below else box_y
    tail_bottom_y = anchor_point[1] if not place_below else box_y

    tail_cx = min(max(anchor_point[0], box_x + 20), box_x + box_w - 20)

    if not place_below:
        triangle = [
            (tail_cx - 14, box_y + box_h),
            (tail_cx + 14, box_y + box_h),
            (anchor_point[0], anchor_point[1])
        ]
    else:
        triangle = [
            (tail_cx - 14, box_y),
            (tail_cx + 14, box_y),
            (anchor_point[0], anchor_point[1])
        ]

    draw.polygon(triangle, fill=(255, 255, 255))
    draw.line(
        [triangle[2], (tail_cx, (triangle[2][1] + box_y) / 2)],
        fill=(255, 0, 0),
        width=3
    )

    # ----- kotak putih -----

    try:
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            radius=14,
            fill=(255, 255, 255),
            outline=(30, 30, 30),
            width=2
        )
    except Exception:
        draw.rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            fill=(255, 255, 255),
            outline=(30, 30, 30),
            width=2
        )

    # ----- teks (hitam, tebal, centered) -----

    text_x = box_x + (box_w - text_w) // 2
    text_y = box_y + padding

    draw.text(
        (text_x, text_y),
        display_text,
        fill=(0, 0, 0),
        font=font
    )

    # ----- crop plat asli -----

    if plate_pil:

        plate_x = box_x + (box_w - plate_pil.width) // 2
        plate_y = text_y + text_h + gap

        canvas.paste(plate_pil, (plate_x, plate_y))

    return canvas


# ==================================================
# MAIN RENDER FUNCTION (dipanggil dari app.py di dalam tab)
# ==================================================

def render():

    st.title("Deteksi dan Pengenalan Plat pada Gambar")

    car_model, plate_model = load_models()
    reader = load_reader()

    # ==================================================
    # FILE UPLOADER
    # ==================================================

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="image_tab_uploader"
    )

    # ==================================================
    # MAIN PIPELINE
    # ==================================================

    if uploaded_file:

        # ==================================================
        # STEP 1 - INPUT IMAGE
        # ==================================================

        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

        st.header("1. Input Image")

        st.image(
            image,
            use_container_width=True
        )

        # ==================================================
        # STEP 2 - VEHICLE DETECTION
        # ==================================================

        st.header("2. Vehicle Detection")

        vehicle_results = car_model(image)[0]

        vehicle_vis = image.copy()

        vehicles = []

        for box in vehicle_results.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            vehicles.append(
                (x1, y1, x2, y2)
            )

            cv2.rectangle(
                vehicle_vis,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        st.image(
            vehicle_vis,
            caption=f"{len(vehicles)} Vehicle(s) Detected"
        )

        # ==================================================
        # STEP 3 - VEHICLE CROPS
        # ==================================================

        st.header("3. Vehicle Crops")

        if len(vehicles) == 0:
            st.warning(
                "No vehicle detected."
            )

        # menampung data untuk visualisasi akhir (step 8)
        final_annotations = []

        for idx, (x1, y1, x2, y2) in enumerate(vehicles):

            vehicle_crop = image[
                y1:y2,
                x1:x2
            ]

            st.subheader(
                f"Vehicle #{idx + 1}"
            )

            st.image(
                vehicle_crop
            )

            # ==================================================
            # STEP 4 - PLATE DETECTION
            # ==================================================

            plate_result = plate_model(
                vehicle_crop
            )[0]

            plate_vis = vehicle_crop.copy()

            # bbox mobil pada crop = bingkai penuh crop ini
            cv2.rectangle(
                plate_vis,
                (0, 0),
                (plate_vis.shape[1] - 1, plate_vis.shape[0] - 1),
                (0, 255, 0),
                3
            )

            plate_boxes = []

            for plate_box in plate_result.boxes:

                px1, py1, px2, py2 = map(
                    int,
                    plate_box.xyxy[0]
                )

                plate_boxes.append(
                    (px1, py1, px2, py2)
                )

                cv2.rectangle(
                    plate_vis,
                    (px1, py1),
                    (px2, py2),
                    (255, 0, 0),
                    2
                )

            st.subheader(
                "4. Plate Detection"
            )

            st.image(
                plate_vis,
                caption=f"{len(plate_boxes)} Plate(s) Detected"
            )

            # ==================================================
            # STEP 5 - PLATE CROPS
            # ==================================================

            st.subheader(
                "5. Plate Crops"
            )

            if len(plate_boxes) == 0:
                st.warning(
                    "No plate detected."
                )

            for p_idx, (px1, py1, px2, py2) in enumerate(plate_boxes):

                original_crop = vehicle_crop[
                    py1:py2,
                    px1:px2
                ]

                w = px2 - px1
                h = py2 - py1

                px1_new = px1 + int(0.03 * w)
                px2_new = px2 - int(0.03 * w)

                py1_new = py1 + int(0.08 * h)
                py2_new = py2 - int(0.35 * h)

                px1_new = max(
                    0,
                    px1_new
                )

                py1_new = max(
                    0,
                    py1_new
                )

                px2_new = min(
                    vehicle_crop.shape[1],
                    px2_new
                )

                py2_new = min(
                    vehicle_crop.shape[0],
                    py2_new
                )

                ocr_crop = vehicle_crop[
                    py1_new:py2_new,
                    px1_new:px2_new
                ]

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        original_crop,
                        caption="Original Detection"
                    )

                with col2:

                    st.image(
                        ocr_crop,
                        caption="OCR Crop"
                    )

                # ==================================================
                # STEP 6 - PREPROCESSING
                # ==================================================

                st.subheader(
                    "6. OCR Preprocessing"
                )

                gray = cv2.cvtColor(
                    ocr_crop,
                    cv2.COLOR_RGB2GRAY
                )

                _, thresh = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                col3, col4 = st.columns(2)

                with col3:

                    st.write(
                        "Grayscale"
                    )

                    st.image(
                        gray,
                        clamp=True
                    )

                with col4:

                    st.write(
                        "Threshold"
                    )

                    st.image(
                        thresh,
                        clamp=True
                    )

                # ==================================================
                # STEP 7 - OCR
                # ==================================================

                st.subheader(
                    "7. OCR Result"
                )

                with st.spinner(
                    "Sedang membaca plat..."
                ):

                    results = reader.readtext(
                        thresh,
                        paragraph=True,
                        width_ths=1.5,
                        x_ths=1.0,
                        y_ths=0.5
                    )

                with st.expander(
                    f"Raw EasyOCR Output - Plate #{p_idx+1}"
                ):
                    st.write(results)

                formatted_plate = None

                if len(results) == 0:

                    st.error(
                        "Tidak ada teks terdeteksi"
                    )

                else:

                    ocr_parts = []

                    for result in results:

                        try:

                            if len(result) == 3:

                                bbox, text, conf = result

                            elif len(result) == 2:

                                bbox, text = result
                                conf = 1.0

                            else:
                                continue

                            cleaned = clean_plate_text(
                                text
                            )

                            if cleaned:

                                ocr_parts.append(
                                    cleaned
                                )

                        except Exception:
                            continue

                    combined_text = ''.join(
                        ocr_parts
                    )

                    st.write(
                        "### OCR Parts"
                    )

                    st.code(
                        str(ocr_parts)
                    )

                    st.write(
                        "### OCR Gabungan"
                    )

                    st.code(
                        combined_text
                    )

                    formatted_plate = (
                        format_indonesian_plate(
                            combined_text
                        )
                    )

                    if formatted_plate:

                        st.success(
                            f"Plat Indonesia Terdeteksi: {formatted_plate}"
                        )

                    else:

                        st.error(
                            "Format plat Indonesia tidak valid"
                        )

                    with st.expander(
                        f"Detail OCR - Plate #{p_idx+1}"
                    ):

                        for result in results:

                            try:

                                if len(result) == 3:

                                    bbox, text, conf = result

                                elif len(result) == 2:

                                    bbox, text = result
                                    conf = 1.0

                                else:
                                    continue

                                st.write(
                                    f"Teks: {text}"
                                )

                                st.write(
                                    f"Confidence: {conf:.2f}"
                                )

                                st.divider()

                            except Exception:
                                pass

                # simpan data untuk ditempel ke gambar asli nanti
                final_annotations.append({
                    "vehicle_box": (x1, y1, x2, y2),
                    "plate_box_full": (
                        x1 + px1, y1 + py1,
                        x1 + px2, y1 + py2
                    ),
                    "plate_crop": original_crop,
                    "plate_text": formatted_plate
                })

        # ==================================================
        # STEP 8 - VISUALISASI AKHIR (ditempel di gambar asli)
        # ==================================================

        st.header("8. Visualisasi Akhir")

        if len(final_annotations) == 0:

            st.warning(
                "Tidak ada anotasi untuk divisualisasikan."
            )

        else:

            annotated = Image.fromarray(image.copy()).convert("RGB")
            draw = ImageDraw.Draw(annotated)
            img_w, img_h = annotated.size

            # gambar dulu semua bbox (corner mobil + kotak plat)
            drawn_vehicle_boxes = set()

            for ann in final_annotations:

                v_box = ann["vehicle_box"]

                if v_box not in drawn_vehicle_boxes:
                    draw_corner_brackets(draw, v_box, color=(150, 255, 0))
                    drawn_vehicle_boxes.add(v_box)

                draw_plate_box(draw, ann["plate_box_full"], color=(255, 0, 0))

            # baru gambar callout (agar tidak ketutup bbox)
            for ann in final_annotations:

                px1f, py1f, px2f, py2f = ann["plate_box_full"]
                anchor = (int((px1f + px2f) / 2), int(py1f))

                annotated = draw_callout(
                    canvas=annotated,
                    anchor_point=anchor,
                    plate_crop=ann["plate_crop"],
                    result_text=ann["plate_text"],
                    image_size=(img_w, img_h)
                )

            st.image(
                np.array(annotated),
                use_container_width=True,
                caption="Hasil Deteksi & OCR pada Gambar Asli"
            )
