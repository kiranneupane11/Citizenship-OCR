import os
from flask import Flask, render_template, request, jsonify
from ocr_core import ocr

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

            # Run OCR and process the extracted text
            ocr_text = ocr(file_path)
            ocr_data = parse_ocr_to_dict(ocr_text)  # Convert OCR text to a dictionary

            # Render the form with dynamically filled values
            return render_template('form.html', ocr_data=ocr_data)

    return render_template('upload.html')


def parse_ocr_to_dict(ocr_text):
    """Parse the OCR text into a dictionary of key-value pairs."""
    lines = ocr_text.split('\n')
    ocr_dict = {}

    # Example parsing logic (you may need to adjust for your OCR output structure)
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            ocr_dict[key.strip()] = value.strip()

    return ocr_dict


if __name__ == '__main__':
    app.run(debug=True)























#             # Call the OCR function
#             extracted_text = ocr(file_path)
#             # Convert extracted text to JSON format
#             extracted_text= jsonify(extracted_text)            

#             # Render the upload page with results
#             return render_template('upload.html',
#                                    msg='Successfully processed',
#                                    extracted_text=extracted_text.get_json(),
#                                    img_src=os.path.join('static', 'uploads', filename))
#     elif request.method == 'GET':
#         return render_template('upload.html')


# if __name__ == '__main__':
#     app.run(debug=True)
