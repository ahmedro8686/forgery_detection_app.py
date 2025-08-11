import streamlit as st
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="Image Forgery Detector", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def load_image_from_upload(ufile):
    file_bytes = np.asarray(bytearray(ufile.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError("Cannot decode image")
    # If 4 channels (RGBA), convert to RGB dropping alpha
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def resize_max_dim(img, max_dim):
    if img is None:
        raise ValueError("Image is None")
    if not hasattr(img, "shape"):
        raise ValueError("Image has no shape attribute")
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def compute_ela(img_rgb, quality=90):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    if not success:
        return np.zeros_like(img_rgb[...,0]).astype(np.float32)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    dec_rgb = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    diff = cv2.absdiff(img_rgb, dec_rgb).astype(np.float32)
    diff_gray = np.mean(diff, axis=2)
    maxv = diff_gray.max() if diff_gray.max() > 0 else 1.0
    ela_norm = diff_gray / maxv
    return ela_norm

def compute_edge_map(img_gray):
    v = np.median(img_gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(img_gray.astype(np.uint8), lower, upper)
    edges = edges.astype(np.float32) / 255.0
    return edges

def compute_lbp(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    padded = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_REFLECT)
    center = padded[1:-1,1:-1]
    bits = np.zeros((h,w,8), dtype=np.uint8)
    neighbors = [
        padded[0:-2,0:-2], padded[0:-2,1:-1], padded[0:-2,2:],
        padded[1:-1,2:], padded[2:,2:], padded[2:,1:-1],
        padded[2:,0:-2], padded[1:-1,0:-2]
    ]
    for i, nb in enumerate(neighbors):
        bits[..., i] = (nb >= center).astype(np.uint8)
    powers = (1 << np.arange(8)).astype(np.uint8)
    lbp = np.sum(bits * powers[::-1], axis=2)
    lbp_norm = lbp.astype(np.float32) / 255.0
    return lbp_norm

def compute_dct_highfreq(img_gray, keep_low=16):
    img_f = img_gray.astype(np.float32) / 255.0
    h, w = img_f.shape
    try:
        dct = cv2.dct(img_f)
        mask = np.ones_like(dct)
        mask[:keep_low, :keep_low] = 0
        dct_high = dct * mask
        idct = cv2.idct(dct_high)
        high = np.abs(idct)
        maxv = high.max() if high.max() > 0 else 1.0
        return high / maxv
    except Exception:
        lap = cv2.Laplacian(img_gray, cv2.CV_32F)
        lap = np.abs(lap)
        maxv = lap.max() if lap.max() > 0 else 1.0
        return lap / maxv

def combine_zscore_map(edges, lbp, noise):
    stack = np.stack([edges, lbp, noise], axis=2).astype(np.float32)
    mu = np.mean(stack, axis=2, keepdims=True)
    sigma = np.std(stack, axis=2, keepdims=True) + 1e-8
    z = (stack - mu) / sigma
    anomaly = np.mean(np.abs(z), axis=2)
    mn, mx = np.min(anomaly), np.max(anomaly)
    den = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    norm = (anomaly - mn) / den
    return norm

def heatmap_to_color(hmap):
    h_uint8 = np.clip((hmap * 255).astype(np.uint8), 0, 255)
    colored = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored_rgb

def overlay_on_image(img_rgb, heat_rgb, alpha=0.45):
    img_u8 = img_rgb.astype(np.uint8)
    heat_u8 = heat_rgb.astype(np.uint8)
    overlay = cv2.addWeighted(img_u8, 1.0 - alpha, heat_u8, alpha, 0)
    return overlay

def threshold_anomaly_map(hmap, method="fixed", fixed_val=0.4):
    if method == "otsu":
        arr = (hmap * 255).astype(np.uint8)
        _, binmap = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (binmap.astype(np.uint8) // 255).astype(np.uint8)
    else:
        binmap = (hmap >= fixed_val).astype(np.uint8)
        return binmap

def find_regions(binary_map):
    bin_u8 = (binary_map * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0.0, 0.0
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append({
            "area": area,
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "contour": cnt
        })
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions

def image_to_bytes(img_rgb, ext=".png"):
    bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(ext, bgr)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()

# ---------------------------
# UI
# ---------------------------
st.title("🔎 Image Forgery Detection — Full App")
st.write("Upload an image and the app will compute ELA, Edge, LBP, DCT noise maps, combine them into a heatmap, and give a final prediction + confidence. (No pretrained CNN — uses analytic fusion + threshold)")

with st.sidebar:
    st.header("Settings")
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

    def norm01(a):
        a = np.nan_to_num(a.astype(np.float32))
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-8:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)
    ela_n = norm01(ela)
    edges_n = norm01(edges)
    lbp_n = norm01(lbp)
    noise_n = norm01(noise)

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
    st.markdown(f"- Prediction: {decision}")
    st.markdown(f"- Confidence (manipulation score): {confidence_score:.2%} (avg heatmap intensity)")
    st.caption("⚠️ Note: This analytic detector flags anomalous regions — for production you should train a supervised classifier (CNN) on labeled real/fake data to get higher reliability.")

    # إضافة رسالة واضحة للحكم النهائي
    st.markdown("---")
    st.header("🛑 Final Authenticity Check")
    if decision == "Fake":
        st.error("⚠️ هذه الصورة **مُزيفة**")
    else:
        st.success("✅ هذه الصورة **حقيقية**")

    img_bytes = image_to_bytes(overlay_annot, ext=".png")
    st.download_button("⬇️ Download overlay (PNG)", data=img_bytes, file_name="anomaly_overlay.png", mime="image/png")

    st.markdown("### Tweak & re-run decision")
    new_thresh = st.slider("Decision threshold (heatmap mean)", 0.01, 0.9, float(fixed_threshold), step=0.01, key="decision_thresh")
    st.write(f"Current mean heatmap score: {manipulation_score:.4f}")
    if st.button("Apply new threshold"):
        new_decision = "Fake" if manipulation_score >= new_thresh else "Real"
        st.success(f"New decision with threshold {new_thresh:.2f}: {new_decision}")

else:
    st.info("Upload an image and click 'Run analysis' to start. If you want a *self-contained* version (from raw image → prediction) I can bundle everything into one file that also includes optional training scaffolding.")

