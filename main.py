import os
from flask import Flask, render_template, request, redirect, send_from_directory, flash
from datetime import datetime

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Part

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_files(folder):
    return sorted(
        [f for f in os.listdir(folder) if allowed_file(f)],
        reverse=True
    )

def analyze_audio_with_gemini(file_path):
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
    model = GenerativeModel("gemini-1.5-pro")
    with open(file_path, "rb") as f:
        audio_part = Part.from_data(data=f.read(), mime_type="audio/wav")

    prompt = """
    Please provide an exact trascript for the audio, followed by sentiment analysis.
    Your response should follow the format:
    Text: USERS SPEECH TRANSCRIPTION
    Sentiment Analysis: Positive | Neutral | Negative
    """

    response = model.generate_content([prompt, audio_part])
    return response.text

@app.route('/')
def index():
    files = get_files(UPLOAD_FOLDER)
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        flash('No audio data')
        return redirect(request.url)

    file = request.files['audio_data']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        result_text = analyze_audio_with_gemini(file_path)
        with open(file_path + '.txt', 'w') as f:
            f.write(result_text)

    return redirect('/')

@app.route('/<folder>/<filename>')
def uploaded_file(folder, filename):
    return send_from_directory(folder, filename)

@app.route('/script.js', methods=['GET'])
def script_js():
    return send_from_directory('', 'script.js')

if __name__ == '__main__':
    app.run(debug=True)
