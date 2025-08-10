# forgery_detection_app.py (Full Version: Illumination + Edges + Texture + Noise Analysis)

import streamlit as st
import cv2
import numpy as np
from matplotlib import cm
from skimage.feature import local_binary_pattern

st.set_page_config(page_title="Image Forgery Detection", layout="wide")

st.title("🕵️ Image Forgery Detection (Full Analysis)")
st.write("Upload an image to detect suspicious regions using illumination, edges, texture, and noise analysis.")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# إعدادات قابلة للتحكم
blur_sigma = st.sidebar.slider("Gaussian Blur Sigma", 1, 100, 30)
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 50, 200, 100)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 100, 300, 200)
anomaly_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.3)
lbp_radius = st.sidebar.slider("LBP Radius", 1, 5, 3)
lbp_points = lbp_radius * 8

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # تحويل للصيغة الصحيحة
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # تحويل لمساحة YCbCr وتحليل قناة الإضاءة Y
    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr_img[:, :, 0]

    # 1️⃣ تحليل الإضاءة
    illumination_map = cv2.GaussianBlur(y_channel.astype(np.float32), (0, 0), blur_sigma)

    # 2️⃣ كشف الحواف
    edges = cv2.Canny(y_channel, canny_thresh1, canny_thresh2)

    # 3️⃣ تحليل القوام باستخدام LBP
    lbp = local_binary_pattern(y_channel, lbp_points, lbp_radius, method="uniform")
    lbp_norm = cv2.normalize(lbp.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    # 4️⃣ تحليل الضوضاء باستخدام DCT
    dct = cv2.dct(np.float32(y_channel) / 255.0)
    dct_abs = np.abs(dct)
    # نأخذ فقط الترددات العالية كخريطة ضوضاء
    h, w = dct_abs.shape
    high_freq = dct_abs[h//4:, w//4:]
    noise_map = cv2.resize(high_freq, (w, h))
    noise_norm = cv2.normalize(noise_map, None, 0, 1, cv2.NORM_MINMAX)

    # دمج التحليلات الأربع
    illum_norm = cv2.normalize(illumination_map, None, 0, 1, cv2.NORM_MINMAX)
    edges_norm = edges.astype(np.float32) / 255.0

    anomaly_map = (illum_norm * 0.3) + (edges_norm * 0.25) + (lbp_norm * 0.25) + (noise_norm * 0.2)

    # تطبيق Threshold لتمييز المناطق الشاذة
    anomaly_binary = (anomaly_map > anomaly_threshold).astype(np.float32)

    # خريطة حرارية
    heatmap = cm.jet(anomaly_map)[:, :, :3]
    overlay = cv2.addWeighted(img_rgb.astype(np.float32)/255.0, 0.6, heatmap, 0.4, 0)

    # عرض النتائج
    st.subheader("🔍 Analysis Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.image(img_rgb, caption="Original Image", use_column_width=True)
    col2.image(illumination_map, caption="Illumination Map (Y Channel)", use_column_width=True, clamp=True)
    col3.image(edges, caption="Edge Detection", use_column_width=True)
    col4.image(lbp, caption="LBP Texture Map", use_column_width=True, clamp=True)

    col5, col6, col7 = st.columns(3)
    col5.image(noise_norm, caption="Noise Map (High Frequencies)", use_column_width=True, clamp=True)
    col6.image(heatmap, caption="Heatmap of Anomalies", use_column_width=True)
    col7.image(overlay, caption="Overlay Heatmap on Original", use_column_width=True)

    st.image(anomaly_binary, caption="Binary Anomaly Regions", use_column_width=True, clamp=True)

    # حفظ النتيجة كصورة للتحميل
    result_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("anomaly_result.png", result_bgr)
    with open("anomaly_result.png", "rb") as file:
        st.download_button("Download Anomaly Map", file, "anomaly_result.png")
