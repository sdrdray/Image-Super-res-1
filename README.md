# Image Super-Resolution Workspace

This repository contains two related image-processing projects:

- `superres6-N`: a Flask web application for image upscaling and denoising.
- `superres6-d`: a small command-line image denoising utility.

The repository also includes research model definitions, configuration files,
examples, and pretrained weights under `superres6-N/models-helaman`.

## Projects

### Web application

The web app supports several pretrained models and exposes a browser interface
for uploading an image, processing it, previewing the result, and downloading
the output. See [superres6-N/README.md](superres6-N/README.md) for setup and
model details.

```text
cd superres6-N
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` after the server starts.

### Command-line denoiser

The lightweight utility uses OpenCV's non-local means denoising algorithm:

```text
cd superres6-d
pip install -r requirements.txt
python denoise.py input.jpg output.jpg
```

## Repository layout

```text
superres6-d/       Command-line denoising tool
superres6-N/       Flask application and model collection
  app.py           Flask API and inference entry point
  index.html       Browser interface
  models-helaman/  Model definitions, configs, and examples
```

## Large files

Pretrained model files are stored with [Git LFS](https://git-lfs.com/). Install
Git LFS before cloning or pulling the repository so the weights are downloaded
instead of placeholder pointer files.

```text
git lfs install
git clone https://github.com/sdrdray/Image-Super-res-1.git
```

Model weights are intended for local inference. Hardware acceleration is
optional; PyTorch falls back to the CPU when CUDA is unavailable.

## Notes

- Keep generated uploads and outputs out of commits unless they are useful
  examples.
- Review the model's license and usage terms before redistributing results.
- The application is intended for local development and experimentation.