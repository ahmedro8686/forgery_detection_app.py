import streamlit as st
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage import measure
from scipy import fftpack

# إعداد واجهة التطبيق
st.title("🕵️ Advanced Image Forgery Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    st.image(original_img, caption="Original Image", use_container_width=True)

    # تحويل للصورة الرمادية
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    gray_f = np.float32(gray) / 255.0

    # -------- Edge Map --------
    edges = cv2.Canny(gray, 100, 200)
    edges_norm = edges / 255.0

    # -------- LBP Texture Map --------
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_norm = lbp / lbp.max()

    # -------- Noise Map (DCT) --------
    dct = fftpack.dct(fftpack.dct(gray_f.T, norm='ortho').T, norm='ortho')
    noise_map = np.abs(dct)
    noise_norm = noise_map / noise_map.max()

    # -------- دمج وتحليل anomalies --------
    combined_map = (edges_norm + lbp_norm + noise_norm) / 3.0
    z_score_map = (combined_map - np.mean(combined_map)) / np.std(combined_map)

    anomaly_binary = (np.abs(z_score_map) > 2).astype(np.uint8)
    regions = measure.regionprops(measure.label(anomaly_binary))

    # -------- رسم الخرائط --------
    cols = st.columns(4)
    cols[0].image(edges_norm, caption="Edge Map", clamp=True, use_container_width=True)
    cols[1].image(lbp_norm, caption="LBP Texture Map", clamp=True, use_container_width=True)
    cols[2].image(noise_norm, caption="Noise Map (DCT)", clamp=True, use_container_width=True)
    cols[3].image(combined_map, caption="Combined Map", clamp=True, use_container_width=True)

    st.image(z_score_map, caption="Z-score Heatmap of Anomalies", use_container_width=True)
    st.image(anomaly_binary, caption="Binary Anomaly Map", clamp=True, use_container_width=True)

    # -------- رسم مستطيلات على المناطق المشبوهة --------
    if len(regions) > 0:
        original_copy = original_img.copy()
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(original_copy, (minc, minr), (maxc, maxr), (255, 0, 0), 2)
            cv2.putText(original_copy, "Suspicious", (minc, minr - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        st.subheader(f"Detected Suspicious Regions: {len(regions)}")
        st.image(original_copy, caption="Suspicious Regions Highlighted", use_container_width=True)
    else:
        st.subheader("No suspicious regions detected.")

    # -------- حساب نسبة الثقة --------
    manipulation_score = float(np.mean(np.abs(z_score_map)))
    threshold = 0.4
    predicted_label = "Fake" if manipulation_score > threshold else "Real"
    st.markdown(f"### 🏷 Prediction: {predicted_label}")
    st.markdown(f"Confidence Score: {manipulation_score:.2%}")

    # -------- حفظ النتيجة --------
    result_bgr = cv2.cvtColor((combined_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite("anomaly_result.png", result_bgr)
    with open("anomaly_result.png", "rb") as file:
        st.download_button("Download Result Image", file, "anomaly_result.png")



