import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Load YOLO11 model
model = YOLO('best.pt')

# Global variable to store last live detection results
live_class_counts = {}
live_class_names = {}

# Initialize camera
camera = cv2.VideoCapture(0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects(image_path):
    # Perform detection with a confidence threshold of 0.3
    results = model.predict(image_path, conf=0.30)
    
    # Get detection results
    result = results[0]
    detected_classes = result.boxes.cls.cpu().numpy()
    class_names = result.names
    
    # Count objects per class
    class_counts = {}
    for cls in detected_classes:
        class_id = int(cls)
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Save the detection image (no color conversion)
    detection_path = os.path.join(app.config['DETECTION_FOLDER'], 'detection_' + os.path.basename(image_path))
    cv2.imwrite(detection_path, result.plot())  # No color conversion
    
    # Create plotly chart (Pie chart)
    if class_counts:
        labels = [class_names[class_id] for class_id in class_counts.keys()]
        values = list(class_counts.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=px.colors.qualitative.Plotly
        )])
        
        fig.update_layout(
            title_text="Object Detection Distribution 🎯",
            title_font_size=20,
            title_x=0.5
        )
        chart = fig.to_html(full_html=False, include_plotlyjs=False, div_id='pie-chart')
        
        # Create Bar chart
        bar_fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=px.colors.qualitative.Plotly
        )])
        
        bar_fig.update_layout(
            title_text="Object Detection Count 📊",
            title_font_size=20,
            title_x=0.5,
            xaxis_title="Object Class",
            yaxis_title="Count"
        )
        bar_chart = bar_fig.to_html(full_html=False, include_plotlyjs=False, div_id='bar-chart')
    else:
        chart = "<p>No objects detected 😞</p>"
        bar_chart = "<p>No objects detected 😞</p>"
    
    return detection_path, class_counts, class_names, chart, bar_chart

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 25  # fallback FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    class_counts = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 5th frame
        if frame_count % 5 == 0:
            results = model.predict(frame, conf=0.50)
            result = results[0]
            detected_classes = result.boxes.cls.cpu().numpy()
            for cls in detected_classes:
                class_id = int(cls)
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            out.write(result.plot())  # Write detected frame
        else:
            out.write(frame)  # Write original frame (no detection overlay)

        frame_count += 1

    cap.release()
    out.release()

    output_path = os.path.join(app.config['DETECTION_FOLDER'], 'detection_' + os.path.basename(video_path))
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename('output.mp4', output_path)

    # Create plotly chart (Bar chart)
    if class_counts:
        labels = [model.names[class_id] for class_id in class_counts.keys()]
        values = list(class_counts.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=px.colors.qualitative.Plotly
        )])

        fig.update_layout(
            title_text="Video Object Detection Distribution 🎯",
            title_font_size=20,
            title_x=0.5
        )
        chart = fig.to_html(full_html=False, include_plotlyjs=False, div_id='pie-chart')

        bar_fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=px.colors.qualitative.Plotly
        )])

        bar_fig.update_layout(
            title_text="Video Object Detection Count 📊",
            title_font_size=20,
            title_x=0.5,
            xaxis_title="Object Class",
            yaxis_title="Count"
        )
        bar_chart = bar_fig.to_html(full_html=False, include_plotlyjs=False, div_id='bar-chart')
    else:
        chart = "<p>No objects detected in video 😞</p>"
        bar_chart = "<p>No objects detected in video 😞</p>"

    return output_path, class_counts, model.names, chart, bar_chart

def gen_frames():
    global live_class_counts, live_class_names
    while True:
        success, frame = camera.read()
        if not success:
            continue  # Try again instead of break
        else:
            # Run YOLO detection on the frame
            results = model.predict(frame, conf=0.40)
            result = results[0]
            detected_classes = result.boxes.cls.cpu().numpy()
            class_names = result.names

            # Count objects per class
            class_counts = {}
            for cls in detected_classes:
                class_id = int(cls)
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

            # Update global live detection results
            live_class_counts = class_counts
            live_class_names = class_names

            # Draw detections on the frame
            detected_frame = result.plot()  # This returns a numpy array

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', detected_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_detect_page')
def image_detect_page():
    return render_template('image_detect.html')

@app.route('/image_detect', methods=['POST'])
def image_detect():
    if 'file' not in request.files:
        return render_template('image_detect.html', result=None)
    file = request.files['file']
    if file.filename == '':
        return render_template('image_detect.html', result=None)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        detection_path, class_counts, class_names, chart, bar_chart = detect_objects(filepath)
        detection_file = url_for('detections', filename=os.path.basename(detection_path))
        result = []
        for class_id, count in class_counts.items():
            result.append({'class_id': class_id, 'class_name': class_names[class_id], 'count': count})
        result.sort(key=lambda x: x['count'], reverse=True)
        return render_template('image_detect.html',
                               original_file=filepath.replace('\\', '/'),
                               detection_file=detection_file,
                               result=result,
                               chart=chart,
                               bar_chart=bar_chart)
    return render_template('image_detect.html', result=None)

@app.route('/video_detect_page')
def video_detect_page():
    return render_template('video_detect.html')

@app.route('/video_detect', methods=['GET', 'POST'])
def video_detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part 😕', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file 😟', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the video
            detection_path, class_counts, class_names, chart, bar_chart = process_video(filepath)
            detection_file = url_for('detections', filename=os.path.basename(detection_path))
            original_file = filepath.replace('\\', '/')
            result = []
            for class_id, count in class_counts.items():
                result.append({'class_id': class_id, 'class_name': class_names[class_id], 'count': count})
            result.sort(key=lambda x: x['count'], reverse=True)

            return render_template(
                'video_detect.html',
                result=result,
                detection_file=detection_file,
                original_file=original_file,
                chart=chart,
                bar_chart=bar_chart
            )
        else:
            flash('Allowed file types are: png, jpg, jpeg, gif, mp4, mov 🚫', 'error')
            return redirect(request.url)
    else:
        # For GET, pass None for all variables
        return render_template(
            'video_detect.html',
            result=None,
            detection_file=None,
            original_file=None,
            chart=None,
            bar_chart=None
        )

@app.route('/download_result/<filename>')
def download_result(filename):
    # Download from detections folder
    return send_from_directory(app.config['DETECTION_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part 😕', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file 😟', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if it's an image or video
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            detection_path, class_counts, class_names, chart, bar_chart = detect_objects(filepath)
            media_type = 'image'
        else:
            detection_path, class_counts, class_names, chart, bar_chart = process_video(filepath)
            media_type = 'video'
        
        detection_file = url_for('detections', filename=os.path.basename(detection_path))
        
        # Prepare results
        results = []
        for class_id, count in class_counts.items():
            results.append({
                'class_id': class_id,
                'class_name': class_names[class_id],
                'count': count
            })
        
        # Sort by count descending
        results.sort(key=lambda x: x['count'], reverse=True)
        
        return render_template('result.html', 
                             original_file=filepath.replace('\\', '/'),
                             detection_file=detection_file,
                             results=results,
                             chart=chart,
                             bar_chart=bar_chart,
                             media_type=media_type,
                             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    else:
        flash('Allowed file types are: png, jpg, jpeg, gif, mp4, mov 🚫', 'error')
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detections/<filename>')
def detections(filename):
    return send_from_directory('static/detections', filename)

@app.route('/live')
def live():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_data')
def live_data():
    # Prepare results for frontend
    results = []
    for class_id, count in live_class_counts.items():
        results.append({
            'class_id': class_id,
            'class_name': live_class_names[class_id] if class_id in live_class_names else str(class_id),
            'count': count
        })
    results.sort(key=lambda x: x['count'], reverse=True)
    return jsonify({'results': results})

@app.route('/live_page')
def live_page():
    return render_template('live_page.html')

if __name__ == '__main__':
    app.run(debug=True)
