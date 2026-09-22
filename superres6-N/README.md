# Image Super-Resolution & Denoising Web App

This project provides a web interface for image super-resolution and denoising using state-of-the-art deep learning models (SwinIR, RRDBNet, custom denoiser). It uses a Flask backend and a modern HTML frontend.

## Features
- Upload images and apply super-resolution or denoising
- Multiple models supported (SwinIR, RRDBNet, custom denoiser)
- Before/after slider and download for results

## Setup Instructions

### 1. Clone the repository and enter the directory

```
git clone <repo-url>
cd superres6-N
```

### 2. Create and activate a Python environment (recommended: conda)

```
conda create -n superres python=3.10
conda activate superres
```

Or use `python -m venv venv` and `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows).

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Download and place model weights

- Place the following files in the project root:
  - `4xRealWebPhoto_v4_dat2.pth`
  - `4xNomos2_hq_atd.pth`
  - `1xDeNoise_realplksr_otf.pth`
- For the 4xNomos8kSC model, use `models-helaman/4xNomos8kSC/4xNomos8kSC_fp32.param` (already present).

### 5. Run the backend server

```
python app.py
```

- The server will start at `http://127.0.0.1:5000/`

### 6. Use the web interface

- Open your browser and go to `http://127.0.0.1:5000/`
- Upload an image, select a model, and click "Upscale Image"
- Download or preview the result

## Troubleshooting
- If you get missing module errors, ensure your environment is activated and dependencies are installed.
- If CUDA is not available, the app will use CPU (slower).
- Make sure model files are in the correct locations as described above.
- If you encounter errors with model loading, check the model path in `app.py` and ensure the file exists.

## Adding New Models
- Add the model weights to the project or a subfolder.
- Update the `model_map` in `app.py` with the new model's configuration.
- Add a new option in `index.html` if you want it selectable from the UI.

## File Structure

```
project-root/
├── app.py
├── index.html
├── requirements.txt
├── README.md
├── 4xRealWebPhoto_v4_dat2.pth
├── 4xNomos2_hq_atd.pth
├── 1xDeNoise_realplksr_otf.pth
├── models-helaman/
│   └── ... (model code and weights)
```

## Credits
- SwinIR, RRDBNet, and other model authors
- This project is a wrapper for research and educational use 