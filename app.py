import streamlit as st

from tabs import image_tab, video_tab

# ==================================================
# STREAMLIT CONFIG
# ==================================================
# set_page_config hanya boleh dipanggil sekali per app,
# jadi dipindah ke sini (dihapus dari masing-masing tab file)

st.set_page_config(
    page_title="Vehicle Detection + Indonesian Plate OCR",
    layout="wide"
)

# ==================================================
# TAB SWITCHER — GAMBAR vs VIDEO
# ==================================================

tab_image, tab_video = st.tabs(["Deteksi Gambar", "Deteksi Video"])

with tab_image:
    image_tab.render()

with tab_video:
    video_tab.render()
