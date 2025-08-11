import streamlit as st
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
from skimage.measure import label, regionprops

# ====== إعداد الصفحة ======
st.set_page_config(page_title="كشف التزييف في الصور", layout="wide")
st.title("🖼️ نظام كشف الصور المزيفة")

# ====== رفع الصورة ======
uploaded_file = st.file_uploader("ارفع صورة للتحليل", type=["jpg", "jpeg", "png"])

# ====== دالة استخراج الخرائط ======
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge Map
    edges = cv2.Canny(gray, 100, 200)
    edges_norm = edges / 255.0

    # LBP Map
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    # Noise Map باستخدام DCT
    dct_img = dct(dct(gray.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')
    noise = np.abs(dct_img)
    noise_norm = (noise - noise.min()) / (noise.max() - noise.min())

    # Z-score anomaly
    z_score_map = (noise_norm - np.mean(noise_norm)) / np.std(noise_norm)
    anomaly_binary = (np.abs(z_score_map) > 2).astype(np.uint8)

    # مناطق مشبوهة
    labeled = label(anomaly_binary)
    regions = regionprops(labeled)

    # Heatmap
    heatmap = cv2.applyColorMap((255 * (np.abs(z_score_map) / np.max(np.abs(z_score_map)))).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay على الصورة الأصلية
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return edges_norm, lbp_norm, noise_norm, heatmap, overlay, anomaly_binary, regions

# ====== دالة التصنيف ======
def classify_image(anomaly_binary, threshold=0.2):
    mean_anomaly = np.mean(anomaly_binary)
    if mean_anomaly > threshold:
        return "Fake", mean_anomaly
    else:
        return "Real", 1 - mean_anomaly

# ====== تنفيذ التحليل ======
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges_norm, lbp_norm, noise_norm, heatmap, overlay, anomaly_binary, regions = process_image(img)

    # تصنيف
    predicted_label, confidence_score = classify_image(anomaly_binary)

    # عرض النتائج
    cols = st.columns(4)
    cols[0].image(edges_norm, caption="Edge Map", clamp=True, use_container_width=True)
    cols[1].image(lbp_norm, caption="LBP Map", clamp=True, use_container_width=True)
    cols[2].image(noise_norm, caption="Noise Map (DCT)", clamp=True, use_container_width=True)
    cols[3].image(overlay, caption="Overlay Heatmap", use_container_width=True)

    st.subheader("Z-score Heatmap")
    st.image(heatmap, caption="Anomaly Heatmap", use_container_width=True)
    st.image(anomaly_binary, caption="Binary Anomaly Map", clamp=True, use_container_width=True)

    st.subheader(f"Detected Suspicious Regions: {len(regions)}")
    for i, region in enumerate(regions, 1):
        st.write(f"Region {i}: Area = {region.area} px, Centroid = ({region.centroid[1]:.1f}, {region.centroid[0]:.1f})")

    st.markdown(f"""
    ### Final Classification Result:
    - Prediction: **{predicted_label}**
    - Confidence Score: **{confidence_score:.2%}**
    """)

    if predicted_label == "Fake":
        st.error("⚠️ هذه الصورة تحتوي على تزييف (مزيفة)")
    else:
        st.success("✅ هذه الصورة أصلية ولا تحتوي على تزييف")

    # زر تحميل النتيجة
    result_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite("anomaly_result.png", result_bgr)
    with open("anomaly_result.png", "rb") as file:
        st.download_button("تحميل صورة النتيجة", file, "anomaly_result.png")

