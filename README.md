Image Forgery Detection App (Python + Streamlit)
📌 Overview

This Python + Streamlit application detects potential forged or manipulated regions in digital images using illumination analysis, edge detection, texture patterns (LBP), and noise frequency mapping.
It integrates multiple computer vision techniques to highlight suspicious areas and visualize anomalies in an interactive way.
🛠 Features

    Illumination Map Analysis – Detects unnatural lighting variations.

    Edge Detection (Canny) – Highlights sharp and unusual boundaries.

    Texture Analysis (LBP) – Finds inconsistencies in surface patterns.

    Noise Analysis (DCT) – Detects abnormal high-frequency patterns.

    Interactive Threshold Control – Adjust sensitivity in real-time.

    Heatmap Overlay – Visual representation of detected anomalies.

    Downloadable Results – Save the processed image with detected regions.

📂 Project Structure

/project-folder
│
├── forgery_detection_app.py   # Main Streamlit app  
├── requirements.txt           # Required Python libraries  
├── sample_image.jpg            # Example input image  
└── README.md                   # Project documentation

🚀 How to Run
Option 1 – Local Execution

    Install Python (>=3.8).

    Install dependencies:

pip install -r requirements.txt

Run the app:

    streamlit run forgery_detection_app.py

    Upload an image in the web interface and adjust settings.

Option 2 – Streamlit Cloud Deployment

    Push your project to a GitHub repository.

    Go to Streamlit Cloud.

    Link your GitHub repo and select forgery_detection_app.py as the entry file.

    Streamlit will auto-install packages from requirements.txt and deploy your app online.

📊 Output Examples

    Illumination Map – Brightness distribution.

    Edge Map – Boundaries of objects.

    Texture Map (LBP) – Pattern differences.

    Noise Map – High-frequency regions.

    Heatmap Overlay – Highlighted suspicious zones.

    Binary Mask – Detected anomaly regions.

📜 License

Released under the MIT License – free to use, modify, and distribute with attribution.
👤 Author

Ahmed Mohamed El-Sayed Mohamed
📧 Email: ahmed.2024zsc@gmail.com

