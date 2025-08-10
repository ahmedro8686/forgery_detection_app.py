import streamlit as st
import cv2
import numpy as np
from matplotlib import cm
from skimage.feature import local_binary_pattern
from scipy.stats import zscore
from skimage.morphology import opening, disk
from skimage.measure import label, regionprops

st.set_page_config(page_title="Advanced Image Forgery Detection", layout="wide")

st.title("🕵️ Advanced Image Forgery Detection")
st.write("Upload an image and detect suspicious regions using multi-technique analysis.")

# --- Sidebar controls ---
blur_sigma = st.sidebar.slider("Gaussian Blur Sigma", 1, 100, 30)
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 50, 200, 100)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 100, 300, 200)
anomaly_threshold = st.sidebar.slider("Anomaly Threshold (z-score)", 0.0, 5.0, 2.0)
lbp_radius = st.sidebar.slider("LBP Radius", 1, 5, 3)
lbp_points = lbp_radius * 8
max_dim = 800  # الحد الأقصى للأبعاد لتحجيم الصور الكبيرة

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def resize_if_large(img, max_dim=800):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def dct_noise_map(y_channel):
    dct = cv2.dct(np.float32(y_channel) / 255.0)
    dct_abs = np.abs(dct)
    h, w = dct_abs.shape
    high_freq = dct_abs[h//4:, w//4:]
    noise_map = cv2.resize(high_freq, (w, h))
    noise_norm = cv2.normalize(noise_map, None, 0, 1, cv2.NORM_MINMAX)
    return noise_norm

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Resize if image too large
    img = resize_if_large(img, max_dim)

    # التعامل مع الصور الرمادية أو الملونة
    if len(img.shape) == 2:
        # صورة رمادية
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        y_channel = img
    elif img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = ycbcr_img[:, :, 0]
    else:
        st.error("Unsupported image format.")
        st.stop()

    # Illumination
    illumination_map = cv2.GaussianBlur(y_channel.astype(np.float32), (0, 0), blur_sigma)
    illum_norm = cv2.normalize(illumination_map, None, 0, 1, cv2.NORM_MINMAX)

    # Edges
    edges = cv2.Canny(y_channel, canny_thresh1, canny_thresh2)
    edges_norm = edges.astype(np.float32) / 255.0

    # Texture (LBP)
    lbp = local_binary_pattern(y_channel, lbp_points, lbp_radius, method="uniform")
    lbp_norm = cv2.normalize(lbp.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    # Noise Analysis (DCT)
    noise_norm = dct_noise_map(y_channel)

    # دمج التحليلات
    combined_map = (illum_norm * 0.3) + (edges_norm * 0.25) + (lbp_norm * 0.25) + (noise_norm * 0.2)

    # حساب z-score
    z_map = zscore(combined_map, axis=None)

    # Threshold binary with morphological opening to remove noise
    anomaly_binary = (z_map > anomaly_threshold).astype(np.uint8)
    anomaly_binary = opening(anomaly_binary, disk(3))

    # استخراج خصائص المناطق المشبوهة
    labeled = label(anomaly_binary)
    regions = regionprops(labeled)

    # Heatmap and Overlay
    heatmap = cm.jet(np.clip(z_map, 0, None))[:, :, :3]
    overlay = cv2.addWeighted(img_rgb.astype(np.float32)/255.0, 0.6, heatmap, 0.4, 0)

    # عرض النتائج
    st.subheader("Original Image")
    st.image(img_rgb, use_column_width=True)

    st.subheader("Individual Analysis Maps")
    cols = st.columns(4)
    cols[0].image(illum_norm, caption="Illumination Map\nShows lighting variations.", clamp=True, use_column_width=True)
    cols[1].image(edges_norm, caption="Edge Map\nDetects sharp boundaries.", clamp=True, use_column_width=True)
    cols[2].image(lbp_norm, caption="Texture Map (LBP)\nAnalyzes local patterns.", clamp=True, use_column_width=True)
    cols[3].image(noise_norm, caption="Noise Map (DCT)\nHighlights high-frequency noise.", clamp=True, use_column_width=True)

    st.subheader("Combined Anomaly Detection")
    st.image(heatmap, caption="Z-score Heatmap of Anomalies", use_column_width=True)
    st.image(overlay, caption="Overlay Heatmap on Original", use_column_width=True)
    st.image(anomaly_binary, caption="Binary Anomaly Map (Z-score Thresholded)", clamp=True, use_column_width=True)

    # عرض ملخص خصائص المناطق المشبوهة
    st.subheader(f"Detected Suspicious Regions: {len(regions)}")
    if len(regions) > 0:
        for i, region in enumerate(regions, 1):
            st.write(f"Region {i}: Area = {region.area} px, Centroid = ({region.centroid[1]:.1f}, {region.centroid[0]:.1f})")

    # حفظ النتيجة
    result_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("anomaly_result.png", result_bgr)
    with open("anomaly_result.png", "rb") as file:
        st.download_button("Download Result Image", file, "anomaly_result.png")


