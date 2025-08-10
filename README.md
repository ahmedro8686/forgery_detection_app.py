Image Forgery Detection App (Python + Streamlit)

📌 Overview

This Python + Streamlit application detects potential forged or manipulated regions in digital images by combining illumination analysis, edge detection, texture patterns (LBP), noise frequency mapping (DCT), and statistical anomaly detection (z-score). It integrates multiple advanced computer vision techniques to highlight suspicious areas and visualize anomalies interactively and intuitively.

🛠 Features

    Illumination Map Analysis – Detects unnatural lighting variations.

    Edge Detection (Canny) – Highlights sharp and unusual boundaries.

    Texture Analysis (Local Binary Patterns - LBP) – Finds inconsistencies in surface patterns.

    Noise Analysis (Discrete Cosine Transform - DCT) – Detects abnormal high-frequency patterns indicative of tampering.

    Statistical Anomaly Detection (Z-score) – Identifies statistically significant irregularities across combined feature maps.

    Interactive Threshold Control – Real-time sensitivity adjustment for anomaly detection.

    Heatmap Overlay – Visual representation of detected anomalies over the original image.

    Downloadable Results – Save the processed images and binary anomaly masks.

    Cross-platform Compatibility – Run locally with Python or deploy easily on Streamlit Cloud.

📂 Project Structure

/project-folder
│
├── forgery_detection_app.py     # Main Streamlit app
├── requirements.txt             # Required Python libraries
├── sample_image.jpg             # Example input image
└── README.md                   # Project documentation

🚀 How to Run

Option 1 – Local Execution

    Install Python (>= 3.8).

    Install dependencies:
    pip install -r requirements.txt

    Run the app:
    streamlit run forgery_detection_app.py

    Upload an image in the web interface and adjust the analysis settings as needed.

Option 2 – Streamlit Cloud Deployment

    Push your project to a GitHub repository.

    Visit Streamlit Cloud.

    Link your GitHub repo and select forgery_detection_app.py as the main file.

    Streamlit Cloud will install dependencies automatically and deploy your app online.

📊 Output Examples

    Illumination Map – Shows brightness distribution.

    Edge Map – Highlights object boundaries.

    Texture Map (LBP) – Displays surface pattern irregularities.

    Noise Map (DCT) – Reveals abnormal frequency components.

    Z-score Heatmap – Statistically weighted anomaly detection.

    Overlay Image – Suspicious areas highlighted over original.

    Binary Anomaly Mask – Final suspicious regions mask.

📜 License

Released under the MIT License – free to use, modify, and distribute with attribution.

👤 Author

Ahmed Mohamed El-Sayed Mohamed
📧 Email: ahmed.2024zsc@gmail.com


