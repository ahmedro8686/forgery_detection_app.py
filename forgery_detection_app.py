import streamlit as st
import numpy as np
import cv2

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
def classify_image(anomaly_binary, z_score_map, pixel_ratio_thresh=0.01, zscore_thresh=3):
    pixel_ratio = np.sum(anomaly_binary) / anomaly_binary.size
    max_zscore = np.max(np.abs(z_score_map))

    if pixel_ratio > pixel_ratio_thresh or max_zscore > zscore_thresh:
        return "Fake", max(pixel_ratio, max_zscore / 10)
    else:
        return "Real", 1 - pixel_ratio

# يفترض أنك محمّل هذه الصور من مرحلة التحليل
# edges_norm, lbp_norm, noise_norm, heatmap, overlay, anomaly_binary, regions

# --- تصنيف ---
predicted_label, confidence_score = classify_image(anomaly_binary, heatmap)

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
