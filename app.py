from flask import Flask, send_from_directory
import subprocess
import sys
import os

app = Flask(__name__)

@app.route('/')
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route('/start')
def start_detection():
    subprocess.Popen([sys.executable, "dds1.py"])
    return "Drowsiness Detection Started! Close the camera window to stop."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
