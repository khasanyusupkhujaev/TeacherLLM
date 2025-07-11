from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from image_processor import extract_text_from_file
from text_corrector import correct_ocr_text
from file_handler import save_text
from homework_check import check_homework
from homework_create import create_homework
from content_create import generate_content

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/homework-check')
def homework_check():
    return render_template('index.html', active_section='homework-check')

@app.route('/homework-create')
def homework_create():
    return render_template('index.html', active_section='homework-create')

@app.route('/content-create')
def content_create():
    return render_template('index.html', active_section='content-create')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', active_section='homework-check', feedback="No file uploaded.", topic="", grade=10, language="uz", formatted_text="")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', active_section='homework-check', feedback="No file selected.", topic="", grade=10, language="uz", formatted_text="")
    if not allowed_file(file.filename):
        return render_template('index.html', active_section='homework-check', feedback="Invalid file type. Allowed: png, jpg, jpeg, pdf.", topic="", grade=10, language="uz", formatted_text="")
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        extracted_text = extract_text_from_file(file_path)
        if not extracted_text:
            return render_template('index.html', active_section='homework-check', feedback="No text extracted from file.", topic="", grade=10, language="uz", formatted_text="")
        corrected_text = correct_ocr_text(extracted_text)
        output_dir = 'images/extracted_text/'
        saved_path = save_text(corrected_text, file_path, output_dir)
        topic = request.form.get('topic', "Algebraik ifodalar")
        grade = int(request.form.get('grade', 10))
        lang = request.form.get('language', "uz")
        result = check_homework(corrected_text, topic, grade, lang)
        if isinstance(result, dict) and "evaluations" in result:
            feedback = f"<p><strong>Corrected Text:</strong><br>{corrected_text.replace(chr(10), '<br>')}</p>"
            feedback += f"<p><strong>Saved to:</strong> {saved_path}</p>"

            for eval in result["evaluations"]:
                feedback += f"""
                <div style='margin-bottom: 1em;'>
                    <p><strong>Problem {eval['problem']}:</strong></p>
                    <p>{eval['feedback'].replace(chr(10), '<br>')}</p>
                    <p><strong>Grade:</strong> {eval['grade']}</p>
                </div>      
                """

            feedback += f"""
            <div style='margin-top: 1em; border-top: 1px solid #ccc; padding-top: 1em;'>
                <p><strong>Overall Feedback:</strong></p>
                <p>{result['overall_feedback'].replace(chr(10), '<br>')}</p>
                <p><strong>Overall Grade:</strong> {result['overall_grade']}</p>
            </div>
            """
            formatted_text = corrected_text.replace('=', ' = ').replace('+', ' + ').replace('-', ' - ').replace('x^2', 'x²').replace('sqrt', '√').replace('\n', '<br>')
        else:
            feedback = f"Corrected Text: {corrected_text}<br>Saved to: {saved_path}<br>Error: {result.get('error', 'Failed to process homework')}"
            formatted_text = corrected_text.replace('\n', '<br>')
        return render_template('index.html', active_section='homework-check', feedback=feedback, topic=topic, grade=grade, language=lang, formatted_text=formatted_text)
    except Exception as e:
        return render_template('index.html', active_section='homework-check', feedback=f"Error processing file: {str(e)}", topic="", grade=10, language="uz", formatted_text="")

@app.route('/create_homework', methods=['POST'])
def create_homework_route():
    topic = request.form.get('create_topic', 'Linear Equations')
    grade = int(request.form.get('create_grade', 8))
    lang = request.form.get('create_language', 'uz')
    number_of_questions = int(request.form.get('number_of_questions', 3))

    homework_content = create_homework(topic, grade, lang, number_of_questions)

    formatted_homework = homework_content.replace('\n', '<br>') if isinstance(homework_content, str) else f"Error: {homework_content}"

    return render_template('index.html', active_section='homework-create', homework_content=formatted_homework, create_topic=topic, create_grade=grade, create_language=lang, number_of_questions=number_of_questions)

@app.route('/create_content', methods=['POST'])
def create_content_route():
    topic = request.form.get('content_topic', 'Elementary Functions')
    grade = int(request.form.get('content_grade', 10))
    lang = request.form.get('content_language', 'uz')

    class_content = generate_content(topic, grade, lang)

    formatted_content = class_content.replace('\n', '<br>') if isinstance(class_content, str) else f"Error: {class_content}"

    return render_template('index.html', active_section='content-create', class_content=formatted_content, content_topic=topic, content_grade=grade, content_language=lang)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('images/extracted_text/'):
        os.makedirs('images/extracted_text/')
    if not os.path.exists('embeddings/'):
        os.makedirs('embeddings/')
    app.run(debug=True)


