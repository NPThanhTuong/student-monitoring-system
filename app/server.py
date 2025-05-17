import subprocess
import sys

from flask import Flask, request, render_template, redirect, send_file

app = Flask(__name__)

# Lưu process để có thể dừng
kafka_process = None
spark_process = None

@app.route('/')
def index():
    kafka_running = kafka_process and kafka_process.poll() is None
    spark_running = spark_process and spark_process.poll() is None
    return render_template('index.html', kafka_running=kafka_running, spark_running=spark_running)

@app.route('/start', methods=['POST'])
def start():
    global kafka_process, spark_process
    python_path = sys.executable
    if kafka_process is None:
        kafka_process = subprocess.Popen([python_path, '-m', 'app.kafka_stream_service'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if spark_process is None:
        spark_process = subprocess.Popen([python_path, '-m', 'app.spark_stream_service'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return redirect('/')

@app.route('/stop', methods=['POST'])
def stop():
    global kafka_process, spark_process

    if kafka_process:
        kafka_process.terminate()
        kafka_process = None
    if spark_process:
        spark_process.terminate()
        spark_process = None

    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
