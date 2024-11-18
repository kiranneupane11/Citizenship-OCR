import os
from flask import Flask, render_template, request
from ocr_core import ocr  # Import your OCR function

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """Check if a file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home_page():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    """Handle file uploads and OCR processing."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')

        file = request.files['file']

        # Check for an empty filename
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')

        # Validate file and process
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Call the OCR function
            extracted_text = ocr(file_path)

            # Render the upload page with results
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   extracted_text=extracted_text,
                                   img_src=os.path.join('static', 'uploads', filename))
    elif request.method == 'GET':
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
