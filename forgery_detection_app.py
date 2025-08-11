import streamlit as st
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage import measure
from scipy.stats import zscore

st.title("🕵️ Advanced Image Forgery Detection")

# ----------- 1. رفع الصورة -----------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # ----------- 2. حساب الخرائط التحليلية -----------
    # Edge Map
    edges = cv2.Canny(bgr_img, 100, 200)
    edges_norm = edges / 255.0

    # LBP Texture Map
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=3, method='uniform')
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    # Noise Map (DCT)
    gray_f = np.float32(gray) / 255.0
    dct = cv2.dct(gray_f)
    high_freq = np.abs(dct)
    noise_norm = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min())

    # ----------- 3. توحيد النتائج باستخدام Z-Score -----------
    combined = (edges_norm + lbp_norm + noise_norm) / 3.0
    heatmap = zscore(combined)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    anomaly_binary = (heatmap > 0.6).astype(np.uint8)

    # ----------- 4. تحديد المناطق المشتبه فيها -----------
    labeled = measure.label(anomaly_binary, connectivity=2)
    regions = measure.regionprops(labeled)

    # ----------- 5. عرض النتائج -----------
    st.image(original_img, caption="Original Image", use_container_width=True)

    cols = st.columns(4)
    cols[0].image(edges_norm, caption="Edge Map", use_container_width=True)
    cols[1].image(lbp_norm, caption="LBP Texture Map", use_container_width=True)
    cols[2].image(noise_norm, caption="Noise Map (DCT)", use_container_width=True)
    cols[3].image(heatmap, caption="Z-score Heatmap", use_container_width=True)

    st.image(anomaly_binary * 255, caption="Binary Anomaly Map", use_container_width=True)

    # رسم المربعات على الأماكن المشتبه بها
    original_copy = original_img.copy()
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(original_copy, (minc, minr), (maxc, maxr), (255, 0, 0), 2)
        cv2.putText(original_copy, "Suspicious", (minc, minr - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    st.image(original_copy, caption="Detected Suspicious Regions", use_container_width=True)

    # ----------- 6. حساب نسبة الثقة -----------
    mean_score = float(np.mean(heatmap))
    area_score = sum([r.area for r in regions]) / (original_img.shape[0] * original_img.shape[1])
    final_score = (mean_score + area_score) / 2

    label = "Fake" if final_score > 0.35 else "Real"

    st.markdown(f"### 🏷 Prediction: {label}")
    st.markdown(f"**Mean Score:** {mean_score:.2%}")
    st.markdown(f"**Area Score:** {area_score:.2%}")
    st.markdown(f"**Final Confidence:** {final_score:.2%}")


