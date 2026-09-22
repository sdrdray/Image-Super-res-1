from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
from werkzeug.utils import secure_filename
import os
import sys
import traceback

# Add SwinIR_main to sys.path
sys.path.append(os.path.abspath('SwinIR_main'))
# Add models-helaman to sys.path
sys.path.append(os.path.abspath('models-helaman'))

import cv2
import torch
import numpy as np
import importlib
import json

RRDBNet = importlib.import_module('network_rrdbnet').RRDBNet
network_denoise_custom = importlib.import_module('network_denoise_custom')
DenoiseCustomNet = network_denoise_custom.DenoiseCustomNet


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def rrdbnet_infer(input_path, output_path, model_path, scale, num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    print(f"[DEBUG] Model path: {model_path}")
    model = RRDBNet(in_nc=num_in_ch, out_nc=num_out_ch, nf=num_feat, nb=num_block, gc=num_grow_ch, scale=scale)
    pretrained_model = torch.load(model_path, map_location=device)
    # Try to load 'params_ema' or 'params' or the whole dict
    param_key_g = 'params_ema' if 'params_ema' in pretrained_model else 'params'
    if isinstance(pretrained_model, dict):
        if param_key_g in pretrained_model:
            model.load_state_dict(pretrained_model[param_key_g], strict=True)
        elif 'state_dict' in pretrained_model:
            model.load_state_dict(pretrained_model['state_dict'], strict=True)
        else:
            model.load_state_dict(pretrained_model, strict=True)
    else:
        model.load_state_dict(pretrained_model, strict=True)
    model.eval()
    model = model.to(device)
    img_lq = cv2.imread(input_path, cv2.IMREAD_COLOR)
    print(f"[DEBUG] Input image shape (HWC): {img_lq.shape}")
    img_lq = img_lq.astype(np.float32) / 255.0
    img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    img_lq = torch.from_numpy(img_lq).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_lq)
    output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    print(f"[DEBUG] Output image numpy shape: {output_img.shape}")
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    print(f"[DEBUG] Output image final shape (HWC): {output_img.shape}")
    output_img = (output_img * 255.0).round().astype(np.uint8)
    cv2.imwrite(output_path, output_img)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def denoise_custom_infer(input_path, output_path, model_path, num_blocks, in_ch, feat_ch, mid_ch):
    import torch
    import cv2
    import numpy as np
    print(f"[DEBUG] denoise_custom_infer called with input_path={input_path}, output_path={output_path}, model_path={model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    model = DenoiseCustomNet(num_blocks=num_blocks, in_ch=in_ch, feat_ch=feat_ch, mid_ch=mid_ch).to(device)
    print("[DEBUG] Model instantiated")
    state = torch.load(model_path, map_location=device)
    print("[DEBUG] Model weights loaded")
    if 'params_ema' in state:
        state_dict = state['params_ema']
        print("[DEBUG] Using params_ema from checkpoint")
    elif 'params' in state:
        state_dict = state['params']
        print("[DEBUG] Using params from checkpoint")
    else:
        state_dict = state
        print("[DEBUG] Using raw state dict from checkpoint")
    model.load_state_dict(state_dict, strict=False)
    print("[DEBUG] Weights loaded into model")
    model.eval()
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    print(f"[DEBUG] Input image shape: {img.shape if img is not None else img}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    print(f"[DEBUG] Input tensor min/max: {img.min()} / {img.max()}")
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"[DEBUG] Input tensor shape: {img.shape}")
    with torch.no_grad():
        out = model(img)
    print(f"[DEBUG] Output tensor min/max before scaling: {out.min().item()} / {out.max().item()}")
    out = out.squeeze().cpu().permute(1, 2, 0).numpy()
    print(f"[DEBUG] Output tensor min/max after squeeze: {out.min()} / {out.max()}")
    # Try saving without multiplying by 255
    out_img = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    print(f"[DEBUG] Output image min/max after scaling: {out_img.min()} / {out_img.max()}")
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_img)
    print(f"[DEBUG] Output image written to {output_path}")

# Update model_map to only include the three models and Helaman
model_map = {
    '4xRealWebPhoto_v4_dat2': {
        'type': 'rrdbnet',
        'model_path': '4xRealWebPhoto_v4_dat2.pth',
        'scale': 4,
        'num_in_ch': 3,
        'num_out_ch': 3,
        'num_feat': 64,
        'num_block': 23,
        'num_grow_ch': 32,
    },
    '4xNomos2_hq_atd': {
        'type': 'rrdbnet',
        'model_path': '4xNomos2_hq_atd.pth',
        'scale': 4,
        'num_in_ch': 3,
        'num_out_ch': 3,
        'num_feat': 64,
        'num_block': 23,
        'num_grow_ch': 32,
    },
    '4xNomos8kSC': {
        'type': 'rrdbnet',
        'model_path': '4xNomos8kSC_fp32.param',
        'scale': 4,
        'num_in_ch': 3,
        'num_out_ch': 3,
        'num_feat': 64,
        'num_block': 23,
        'num_grow_ch': 32,
    },
    '1xDeNoise_realplksr_otf': {
        'type': 'denoise_custom',
        'model_path': '1xDeNoise_realplksr_otf.pth',
        'num_blocks': 30,
        'in_ch': 3,
        'feat_ch': 64,
        'mid_ch': 128
    },
}

@app.route('/process', methods=['POST'])
def process_image():
    print("[DEBUG] /process endpoint called")
    if 'image' not in request.files:
        print("[DEBUG] No image in request.files")
        return jsonify({'success': False, 'error': 'No image uploaded.'})
    file = request.files['image']
    if not file or not file.filename:
        print("[DEBUG] No selected file")
        return jsonify({'success': False, 'error': 'No selected file.'})
    if not allowed_file(file.filename):
        print("[DEBUG] Unsupported file type")
        return jsonify({'success': False, 'error': 'Unsupported file type.'})
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)
    print(f"[DEBUG] File saved to {input_path}")
    model_choice = request.form.get('model_choice', '4xRealWebPhoto_v4_dat2')
    print(f"[DEBUG] model_choice: {model_choice}")
    if model_choice not in model_map:
        print("[DEBUG] Invalid model choice")
        return jsonify({'success': False, 'error': 'Invalid model choice.'})
    model_info = model_map[model_choice]
    output_filename = f"output_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    try:
        if model_info.get('type') == 'rrdbnet':
            print("[DEBUG] rrdbnet_infer selected (skipping for now)")
            rrdbnet_infer(
                input_path, output_path, model_info['model_path'], model_info['scale'],
                model_info['num_in_ch'], model_info['num_out_ch'], model_info['num_feat'],
                model_info['num_block'], model_info['num_grow_ch']
            )
        elif model_info['type'] == 'denoise_custom':
            print("[DEBUG] denoise_custom_infer selected")
            denoise_custom_infer(
                input_path, output_path,
                model_info['model_path'],
                model_info['num_blocks'],
                model_info['in_ch'],
                model_info['feat_ch'],
                model_info['mid_ch']
            )
            print("[DEBUG] denoise_custom_infer completed")
            return jsonify({'success': True, 'message': 'Image processed successfully!', 'output_filename': output_filename})
        else:
            print("[DEBUG] Unknown model type")
            return jsonify({'success': False, 'error': 'Unknown model type.'})
        print("[DEBUG] rrdbnet_infer completed")
        return jsonify({'success': True, 'message': 'Image processed successfully!', 'output_filename': output_filename})
    except Exception as e:
        print("[DEBUG] Exception occurred in /process endpoint")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 