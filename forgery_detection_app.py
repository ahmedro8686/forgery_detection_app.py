import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import layers, models

# --- (احتفظ بباقي الدوال كما هي) ---

# دالة لبناء نموذج CNN بسيط
def build_cnn(input_shape=(128,128,4)):
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# دالة لتحضير بيانات التدريب البسيطة (توليدية)
def generate_synthetic_training_data(num_samples=100):
    # صور حقيقية: خرائط تشوه منخفضة
    real_samples = np.random.rand(num_samples//2, 128, 128, 4)*0.3
    real_labels = np.zeros((num_samples//2, 1))
    # صور مزيفة: خرائط تشوه مرتفعة
    fake_samples = np.random.rand(num_samples//2, 128, 128, 4)*0.7 + 0.3
    fake_labels = np.ones((num_samples//2, 1))
    X = np.concatenate([real_samples, fake_samples], axis=0)
    y = np.concatenate([real_labels, fake_labels], axis=0)
    # خلط البيانات
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    return X[idx], y[idx]

# دالة لتحويل خرائط التحليل إلى مدخلات النموذج (4 قنوات)
def prepare_cnn_input(ela, edges, lbp, noise):
    from cv2 import resize
    def norm_resize(a):
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        return cv2.resize(a, (128,128), interpolation=cv2.INTER_AREA)
    ela_r = norm_resize(ela)
    edges_r = norm_resize(edges)
    lbp_r = norm_resize(lbp)
    noise_r = norm_resize(noise)
    stacked = np.stack([ela_r, edges_r, lbp_r, noise_r], axis=-1)
    return np.expand_dims(stacked, axis=0).astype(np.float32)

# تحميل أو بناء وتدريب النموذج مرة واحدة
@st.cache_resource
def get_trained_cnn():
    model = build_cnn()
    X_train, y_train = generate_synthetic_training_data(200)
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    return model

# --- واجهة المستخدم والإعدادات تبقى كما هي ---

st.title("🔎 Image Forgery Detection — Full App with CNN Enhancement")
st.write("Upload an image and the app computes analytical maps and also uses a simple CNN model for improved forgery detection.")

with st.sidebar:
    max_dim = st.slider("Max image dimension (px)", 256, 2048, 1024, step=64)
    ela_quality = st.slider("ELA JPEG quality (recompress)", 50, 95, 90)
    dct_low = st.slider("DCT low-frequency block size (smaller → more HF)", 4, 128, 16)
    threshold_method = st.selectbox("Binary threshold method", ["fixed", "otsu"])
    fixed_threshold = st.slider("Fixed heatmap threshold (used if method=fixed)", 0.01, 0.9, 0.35, step=0.01)
    overlay_alpha = st.slider("Overlay alpha", 0.05, 0.9, 0.45, step=0.05)
    min_region_area = st.number_input("Min region area (px) to report", 0, 1000000, 20, step=10)

uploaded = st.file_uploader("Upload an image (PNG/JPG/GIF...)", type=["png","jpg","jpeg","bmp","gif"])
run_btn = st.button("Run analysis")

if uploaded is not None and run_btn:
    try:
        img_rgb = load_image_from_upload(uploaded)
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    img_rgb = resize_max_dim(img_rgb, max_dim)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    with st.spinner("Computing ELA, Edge map, LBP, DCT noise..."):
        ela = compute_ela(img_rgb, quality=ela_quality)
        edges = compute_edge_map(img_gray)
        lbp = compute_lbp(img_gray)
        noise = compute_dct_highfreq(img_gray, keep_low=dct_low)

    ela_n = (ela - ela.min()) / (ela.max() - ela.min() + 1e-8)
    edges_n = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    lbp_n = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
    noise_n = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    heatmap = combine_zscore_map(edges_n, lbp_n, noise_n)

    if threshold_method == "otsu":
        binary = threshold_anomaly_map(heatmap, method="otsu")
    else:
        binary = threshold_anomaly_map(heatmap, method="fixed", fixed_val=fixed_threshold)

    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)

    regions = find_regions(binary)
    regions = [r for r in regions if r["area"] >= min_region_area]

    heat_color = heatmap_to_color(heatmap)
    overlay = overlay_on_image(img_rgb, heat_color, alpha=overlay_alpha)

    overlay_annot = overlay.copy()
    for i, r in enumerate(regions, 1):
        x, y, w, h = r["bbox"]
        cv2.rectangle(overlay_annot, (x, y), (x+w, y+h), (255,255,255), 2)
        cx, cy = int(r["centroid"][0]), int(r["centroid"][1])
        cv2.putText(overlay_annot, f"{i}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    manipulation_score = float(np.mean(heatmap))
    confidence_score = manipulation_score
    decision_threshold = fixed_threshold
    decision = "Fake" if manipulation_score >= decision_threshold else "Real"

    # استخدم نموذج CNN للتوقع
    model = get_trained_cnn()
    cnn_input = prepare_cnn_input(ela_n, edges_n, lbp_n, noise_n)
    cnn_pred = model.predict(cnn_input)[0][0]
    cnn_decision = "Fake" if cnn_pred >= 0.5 else "Real"

    st.subheader("Analytical Maps")
    cols = st.columns(4)
    cols[0].image((ela_n * 255).astype(np.uint8), caption="ELA Map", use_container_width=True)
    cols[1].image((edges_n * 255).astype(np.uint8), caption="Edge Map", use_container_width=True)
    cols[2].image((lbp_n * 255).astype(np.uint8), caption="LBP Texture Map", use_container_width=True)
    cols[3].image((noise_n * 255).astype(np.uint8), caption="DCT High-frequency Map", use_container_width=True)

    st.subheader("Combined Anomaly Detection")
    c1, c2 = st.columns([1,1])
    with c1:
        st.image((heat_color).astype(np.uint8), caption="Z-score Heatmap (colored)", use_container_width=True)
        st.image((binary*255).astype(np.uint8), caption="Binary Anomaly Map", use_container_width=True)
    with c2:
        st.image(overlay_annot.astype(np.uint8), caption="Overlay + Detected Regions", use_container_width=True)
        st.markdown(f"Detected suspicious regions (area >= {min_region_area} px): {len(regions)}")
        if len(regions) > 0:
            for i, r in enumerate(regions, 1):
                cx, cy = r["centroid"]
                st.write(f"- Region {i}: Area = {r['area']:.1f} px, Centroid = ({cx:.1f}, {cy:.1f})")

    st.markdown("---")
    st.subheader("Final Result")
    st.markdown(f"- Analytical Prediction: {decision} (Confidence: {confidence_score:.2%})")
    st.markdown(f"- CNN Model Prediction: {cnn_decision} (Confidence: {cnn_pred:.2%})")

    st.markdown("---")
    st.header("🛑 Final Authenticity Check")
    if decision == "Fake" or cnn_decision == "Fake":
        st.error("⚠️ هذه الصورة **مُزيفة**")
    else:
        st.success("✅ هذه الصورة **حقيقية**")

    img_bytes = image_to_bytes(overlay_annot, ext=".png")
    st.download_button("⬇️ Download overlay (PNG)", data=img_bytes, file_name="anomaly_overlay.png", mime="image/png")

else:
    st.info("Upload an image and click 'Run analysis' to start.")

