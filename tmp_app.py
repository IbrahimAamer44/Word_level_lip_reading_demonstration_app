from flask import Flask, render_template
from datetime import datetime
import time

app = Flask(__name__)

def get_time():
    return datetime.now().strftime("%H:%M:%S")

@app.route('/')
def index():
    return render_template('tmp.html')

@app.route('/time')
def time_endpoint():
    def generate():
        while True:
            yield f"data: {get_time()}\n\n"
            time.sleep(1)

    return app.response_class(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
