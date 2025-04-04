from flask import Flask, Response, render_template, request
from traffic import process_traffic_feed
from crowd import process_crowd_feed

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/incident-log')
def incident_log():
    return render_template('incident_log.html')

@app.route('/detections')
def detections():
    return render_template('detections.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/traffic_feed')
def traffic_feed():
    """Serves the raw traffic video feed (no AI processing)."""
    detect = request.args.get('detect', 'false').lower() == 'true'
    return Response(process_traffic_feed("D:\VIT\openCV\dashboard\sample3.mp4", detect=detect), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/crowd_feed')
def crowd_feed():
    """Serves the raw crowd video feed (no AI processing)."""
    detect = request.args.get('detect', 'false').lower() == 'true'
    return Response(process_crowd_feed(detect=detect), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
