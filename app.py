import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from pipeline import CataractAnalysisPipeline
from dataset import NuclearCataractDatabase

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Initialize End-to-End Pipeline
pipeline = CataractAnalysisPipeline()

# Load sample metadata if available
SAMPLE_CACHE = []
def init_samples():
    global SAMPLE_CACHE
    try:
        ds = NuclearCataractDatabase()
        for item in ds.samples[:20]:
            img_bgr = cv2.imread(item['path'])
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                thumb = cv2.resize(img_rgb, (160, 160))
                b64 = pipeline.array_to_base64(thumb)
                SAMPLE_CACHE.append({
                    'filename': item['filename'],
                    'path': item['path'],
                    'label': item['label'],
                    'class_name': item['class_name'],
                    'thumbnail': f"data:image/png;base64,{b64}"
                })
    except Exception as e:
        print(f"Error caching dataset samples: {e}")
            
init_samples()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples', methods=['GET'])
def get_samples():
    return jsonify({'status': 'success', 'samples': SAMPLE_CACHE})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        img_rgb = None
        
        # Check if file upload
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_rgb = np.array(pil_img)
            
        # Check if sample path or base64 provided
        elif request.is_json:
            data = request.get_json()
            if 'sample_path' in data:
                sample_path = data['sample_path']
                if os.path.exists(sample_path):
                    img_bgr = cv2.imread(sample_path)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elif 'image_b64' in data:
                b64_str = data['image_b64'].split(',')[-1]
                img_bytes = base64.b64decode(b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_rgb = np.array(pil_img)

        if img_rgb is None:
            return jsonify({'status': 'error', 'message': 'No valid image provided'}), 400

        # Run End-to-End Pipeline
        results = pipeline.analyze_image(img_rgb)
        return jsonify({'status': 'success', 'data': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("Launching Nuclear Cataract XAI Web Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
