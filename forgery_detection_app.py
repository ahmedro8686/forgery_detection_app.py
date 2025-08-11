import streamlit as st
import numpy as np
import cv2
import traceback

# --- دالة استخراج الميزات الأساسية من الخرائط ---
def extract_features(edges, lbp, noise):
    features = []
    features.append(np.mean(edges))
    features.append(np.std(edges))
    features.append(np.mean(lbp))
    features.append(np.std(lbp))
    features.append(np.mean(noise))
    features.append(np.std(noise))
    return np.array(features)

# --- مثال: حساب النتيجة من خريطة الـ anomalies ---
def classify_image(anomaly_binary, threshold=0.2):
    mean_anomaly = np.mean(anomaly_binary)
    if mean_anomaly > threshold:
        return "Fake", mean_anomaly
    else:
        return "Real", 1 - mean_anomaly

# --- محاولة تحميل البيانات الفعلية أو توليد بيانات وهمية ---
try:
    # جرب تحميل المتغيرات من ملفات (عدل حسب ملفاتك)
    edges_norm = np.load("edges_norm.npy")
    lbp_norm = np.load("lbp_norm.npy")
    noise_norm = np.load("noise_norm.npy")
    heatmap = cv2.imread("heatmap.png") / 255.0
    overlay = cv2.imread("overlay.png") / 255.0
    anomaly_binary = np.load("anomaly_binary.npy")
    regions = []  # يمكن تحمّل بيانات المناطق من ملف pickle إذا متوفر
except Exception as e:
    st.error("حدث خطأ أثناء تحميل البيانات، سيتم استخدام بيانات وهمية للتجربة.")
    st.error(traceback.format_exc())
    # لو الملفات مش موجودة أو فيه خطأ، نولّد بيانات وهمية للتجربة
    edges_norm = np.random.rand(256, 256)
    lbp_norm = np.random.rand(256, 256)
    noise_norm = np.random.rand(256, 256)
    heatmap = np.random.rand(256, 256, 3)
    overlay = np.random.rand(256, 256, 3)
    anomaly_binary = np.random.choice([0, 1], size=(256, 256), p=[0.8, 0.2])
    regions = []

# --- تصنيف ---
predicted_label, confidence_score = classify_image(anomaly_binary)

# --- واجهة Streamlit ---
st.markdown("""
### About the Detection Model
This application uses:
- Edge Map (sharp boundaries detection)
- LBP (texture patterns)
- DCT (noise analysis)
- Z-score anomaly heatmap

The combined results highlight likely manipulated areas.
---
""")

# عرض الصور
cols = st.columns(4)
cols[0].image(edges_norm, caption="Edge Map\nDetects sharp boundaries.", clamp=True, use_container_width=True)
cols[1].image(lbp_norm, caption="Texture Map (LBP)\nAnalyzes local patterns.", clamp=True, use_container_width=True)
cols[2].image(noise_norm, caption="Noise Map (DCT)\nHighlights high-frequency noise.", clamp=True, use_container_width=True)
cols[3].image(overlay, caption="Overlay Heatmap on Original", use_container_width=True)

st.subheader("Combined Anomaly Detection")
st.image(heatmap, caption="Z-score Heatmap of Anomalies", use_container_width=True)
st.image(anomaly_binary, caption="Binary Anomaly Map (Z-score Thresholded)", clamp=True, use_container_width=True)

# ملخص المناطق
st.subheader(f"Detected Suspicious Regions: {len(regions)}")
if len(regions) > 0:
    for i, region in enumerate(regions, 1):
        st.write(f"Region {i}: Area = {region.area} px, Centroid = ({region.centroid[1]:.1f}, {region.centroid[0]:.1f})")

# النتيجة النهائية
st.markdown(f"""
### Final Classification Result:
- Prediction: {predicted_label}
- Confidence Score: {confidence_score:.2%}
""")

# عرض جملة مباشرة إذا الصورة مزيفة أو أصلية
if predicted_label == "Fake":
    st.error("⚠️ هذه الصورة تحتوي على تزييف (مزيفة)")
else:
    st.success("✅ هذه الصورة أصلية ولا تحتوي على تزييف")

# حفظ وتحميل الصورة
result_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("anomaly_result.png", result_bgr)
with open("anomaly_result.png", "rb") as file:
    st.download_button("Download Result Image", file, "anomaly_result.png")

st.info("""
Performance Tips:
- Use resized images for faster processing.
- Adjust thresholds for better accuracy.
""")

