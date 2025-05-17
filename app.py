from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import os
import cv2
import torch
from ultralytics import YOLO
import shutil
import logging
import base64
import numpy as np
from io import BytesIO
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'Model/best.pt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Email configuration for authentication
SMTP_EMAIL = 'saptarshi0777@gmail.com'  # Your Gmail account for SMTP authentication
EMAIL_PASSWORD = 'rwif bbkf hmrw mcor'  # Your Gmail App Password
EMAIL_RECEIVER = 'saptarshi0777@gmail.com'  # Receiver for reports

# Setup logging with DEBUG level for detailed output
logging.basicConfig(level=logging.DEBUG)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

def cleanup_folders():
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def send_email(name, user_email, message):
    """Send report email to saptarshi0777@gmail.com and a reply to the user."""
    smtp_server = None
    try:
        # Initialize SMTP connection
        smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
        smtp_server.starttls()
        smtp_server.login(SMTP_EMAIL, EMAIL_PASSWORD)
        logging.debug("SMTP login successful")

        # 1. Send the report email to saptarshi0777@gmail.com
        subject_report = f"SafeWork Safety Report from {name}"
        body_report = f"Name: {name}\nEmail: {user_email}\nMessage: {message}"
        msg_report = MIMEText(body_report)
        msg_report['Subject'] = subject_report
        msg_report['From'] = SMTP_EMAIL  # Use authenticated email as sender
        msg_report['To'] = EMAIL_RECEIVER
        msg_report['Reply-To'] = user_email  # Replies go to the user

        smtp_server.send_message(msg_report)
        logging.info(f"Report email sent successfully to {EMAIL_RECEIVER} with Reply-To: {user_email}")

        # 2. Send a greeting reply email to the user
        subject_reply = "Thank You for Your SafeWork Report!"
        body_reply = (
            f"Dear {name},\n\n"
            "Thank you for submitting your safety report to SafeWork. We’ve received your message and will review it soon. "
            "Your input helps us ensure safer work environments for everyone.\n\n"
            "If you have any further questions or concerns, feel free to reply to this email.\n\n"
            "Best regards,\n"
            "The SafeWork Team"
        )
        msg_reply = MIMEText(body_reply)
        msg_reply['Subject'] = subject_reply
        msg_reply['From'] = SMTP_EMAIL  # Sent from your Gmail account
        msg_reply['To'] = user_email
        msg_reply['Reply-To'] = SMTP_EMAIL  # Replies go back to SafeWork

        smtp_server.send_message(msg_reply)
        logging.info(f"Reply email sent successfully to {user_email}")

    except smtplib.SMTPRecipientsRefused as e:
        logging.error(f"Recipient refused: {str(e)} - Failed to send to {user_email}")
        raise Exception(f"Failed to send reply email: Invalid or rejected email address {user_email}")
    except smtplib.SMTPAuthenticationError as e:
        logging.error(f"SMTP authentication error: {str(e)}")
        raise Exception("SMTP authentication failed. Check your email and App Password.")
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error: {str(e)}")
        raise Exception(f"SMTP error occurred: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        if smtp_server:
            smtp_server.quit()
            logging.debug("SMTP connection closed")

@app.route('/')
def index():
    cleanup_folders()
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']  # User's email from form
        message = request.form['message']
        logging.info(f"Report received: {name}, {email}, {message}")
        
        try:
            send_email(name, email, message)
            return jsonify({"message": "Report submitted successfully! Check your email for confirmation."}), 200
        except Exception as e:
            logging.error(f"Report submission failed: {str(e)}")
            return jsonify({"error": "Report not submitted, try again."}), 500
            
    return render_template('report.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            results = model(file_path)
            result_img_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + filename)
            results[0].save(result_img_path)
            return jsonify({"image": f"/results/result_{filename}"})

        elif filename.endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return jsonify({"error": "Failed to open video"}), 500
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            output_video_path = os.path.join(app.config['RESULT_FOLDER'], 'output_' + filename.rsplit('.', 1)[0] + '.mp4')
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            cap.release()
            out.release()
            if os.path.exists(output_video_path):
                return jsonify({"video": f"/results/output_{filename.rsplit('.', 1)[0]}.mp4"})
            else:
                return jsonify({"error": "Video processing failed"}), 500

        else:
            return jsonify({"error": "Invalid file format"}), 400

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    logging.info("Client connected to WebSocket")

@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected from WebSocket")

@socketio.on('frame')
def handle_frame(data):
    try:
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("Failed to decode frame")
            return

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            logging.error("Failed to encode processed frame")
            return

        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        processed_data = f'data:image/jpeg;base64,{frame_base64}'
        emit('processed_frame', processed_data)
    except Exception as e:
        logging.error(f"Error processing frame: {str(e)}")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)