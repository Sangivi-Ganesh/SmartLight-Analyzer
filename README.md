# 💡 SmartLight Analyzer

**Real-Time Image Enhancement Using Histogram Equalization**

## About the Project

SmartLight Analyzer is a Python-based image enhancement project developed as part of the Digital Image Processing (ECE3401) CA-3.

It analyzes images captured under different lighting conditions and applies image enhancement techniques to improve visibility and contrast.

## Features

- Image preprocessing and grayscale conversion
- Manual Global Histogram Equalization
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Gamma Correction
- YCrCb-based colour enhancement
- Image histogram analysis
- Performance evaluation using mean intensity, standard deviation and entropy
- Interactive Streamlit dashboard
- Image upload and enhancement

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Plotly
- Streamlit

## Project Structure

```text
SmartLight-Analyzer/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── core/
├── dataset/
├── outputs/
└── reports/
```

## Running Locally

1. Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate into the project folder:

```bash
cd SmartLight-Analyzer
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run app.py
```

## Enhancement Techniques

1. Global Histogram Equalization
2. CLAHE
3. Gamma Correction
4. YCrCb + CLAHE Colour Enhancement

## Dataset

The project uses self-captured images under four illumination conditions:

- Day
- Night
- White Light
- Warm Light

## Author

Digital Image Processing CA-3 Project  
ECE3401 | Academic Year 2026–2027