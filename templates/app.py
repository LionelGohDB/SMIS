import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, flash, make_response, Response, session
from flask_bcrypt import Bcrypt
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import os
from werkzeug.utils import secure_filename
import csv
import io
from io import StringIO  
import matplotlib.pyplot as plt
import base64
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'secret_key'  # Change this to a secure secret key
bcrypt = Bcrypt(app)
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Firebase Initialization
cred = credentials.Certificate('smis-10342-firebase-adminsdk-63ine-c67f217f30.json')  # Add your Firebase service account JSON here
firebase_admin.initialize_app(cred)
db = firestore.client()

# Function to check if uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to check if user is logged in
def is_logged_in():
    return 'user' in session
    
# Home route
@app.route('/')
def index():
    return render_template('index.html', user=session.get('user'))

# Route to register a new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Create user profile in Firestore
        user_profile = {
            'username': username,
            'password': hashed_password
        }
        db.collection('users').document(username).set(user_profile)
        return redirect(url_for('login'))

    return render_template('register.html')

# Route to login an existing user
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_doc = db.collection('users').document(username).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            if bcrypt.check_password_hash(user_data['password'], password):
                session['user'] = username
                return redirect(url_for('index'))
        return 'Invalid username or password'

    return render_template('login.html')


# Route to logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/user-data')
def user_data():
    if 'user' in session:
        user_doc = db.collection('users').document(session['user']).get()
        user_profile = user_doc.to_dict()
        return render_template('user_data.html', user_profile=user_profile)
    return redirect(url_for('login'))

# Upload Excel File Route (categorized by class)
@app.route('/upload-student-data', methods=['GET', 'POST'])
def upload_student_data():
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        student_class = request.form['student_class']
        class_exists = db.collection('students').where('class', '==', student_class).limit(1).get()

        if class_exists:
            flash(f'Class {student_class} already exists. Do you want to replace it?', 'warning')
            return redirect(request.url)
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                df = pd.read_excel(file_path)
                if set(['name', 'class', 'identity_number', 'tahap_penguasaan_1', 'tahap_penguasaan_2']).issubset(df.columns):
                    for index, row in df.iterrows():
                        student_data = {
                            'name': row['name'],
                            'class': row['class'],
                            'identity_number': row['identity_number'],
                            'tahap_penguasaan_1': row['tahap_penguasaan_1'],
                            'tahap_penguasaan_2': row['tahap_penguasaan_2']
                        }
                        db.collection('students').add(student_data)
                    flash('File successfully uploaded and data stored in Firestore', 'success')
                else:
                    flash('Excel file must contain columns: name, class, identity_number, tahap_penguasaan_1, tahap_penguasaan_2', 'danger')
            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
            return redirect(url_for('upload_student_data'))

    return render_template('upload.html')

# Route to add students
@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        student_data = {
            'name': request.form['name'],
            'class': request.form['class'],
            'identity_number': request.form['identity_number'],
            'tahap_penguasaan_1': None,  # Initially set to None
            'tahap_penguasaan_2': None   # Initially set to None
        }
        db.collection('students').add(student_data)  # Add student data to Firestore
        flash('Student added successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('student_form.html')  # Ensure this points to your add student form

# Route to display all students with class filter
@app.route('/students', methods=['GET', 'POST'])
def students():
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    students_ref = db.collection('students')
    if request.method == 'POST':
        class_filter = request.form.get('class_filter')
        if class_filter:
            students_ref = students_ref.where('class', '==', class_filter)

    docs = students_ref.stream()
    student_list = [{**doc.to_dict(), 'id': doc.id} for doc in docs]
    unique_classes = {student['class'] for student in student_list}

    return render_template('students.html', students=student_list, unique_classes=unique_classes)

# Route to update student's score (Tahap Penguasaan)
@app.route('/update-score/<student_id>', methods=['POST'])
def update_score(student_id):
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            tahap_penguasaan_1 = int(request.form['tahap_penguasaan_1'])
            tahap_penguasaan_2 = int(request.form['tahap_penguasaan_2'])

            # Validate tahap_penguasaan range
            if not (1 <= tahap_penguasaan_1 <= 6) or not (1 <= tahap_penguasaan_2 <= 6):
                flash('Tahap Penguasaan must be between 1 and 6.', 'danger')
                return redirect(url_for('students'))

            # Update the Firestore document
            db.collection('students').document(student_id).update({
                'tahap_penguasaan_1': tahap_penguasaan_1,
                'tahap_penguasaan_2': tahap_penguasaan_2
            })
            flash('Score updated successfully!', 'success')
        except Exception as e:
            flash(f'Error updating score: {e}', 'danger')
        return redirect(url_for('students'))

# Route to delete a student
@app.route('/delete-student/<student_id>', methods=['POST'])
def delete_student(student_id):
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    try:
        db.collection('students').document(student_id).delete()
        flash('Student deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting student: {e}', 'danger')
    return redirect(url_for('students'))

# Route to delete all students in a class
@app.route('/delete-class/<student_class>', methods=['POST'])
def delete_class(student_class):
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    try:
        students_ref = db.collection('students').where('class', '==', student_class).stream()
        for student in students_ref:
            db.collection('students').document(student.id).delete()
        flash(f'All students in class {student_class} deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting class: {e}', 'danger')
    return redirect(url_for('students'))


# Route to download students list in a class
@app.route('/download-students', methods=['POST'])
def download_students():
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    class_filter = request.form.get('class_download_filter', '')

    # Fetch students from Firestore based on the class filter
    if class_filter:
        students_ref = db.collection('students').where('class', '==', class_filter)
    else:
        students_ref = db.collection('students')
    
    docs = students_ref.stream()
    student_list = [doc.to_dict() for doc in docs]

    # Create an in-memory output file for the CSV
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'Identity Number', 'Class', 'Tahap Penguasaan 1', 'Tahap Penguasaan 2'])  # Updated header
    for student in student_list:
        cw.writerow([
            student['name'], student['identity_number'], student['class'], student['tahap_penguasaan_1'], student['tahap_penguasaan_2']   # Access the correct variable names
        ])

    # Get CSV content as string
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=students_{class_filter or 'all_classes'}.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output

# Route to display analytics
@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    students_ref = db.collection('students')
    
    # Fetch all unique classes for the dropdown
    all_students = students_ref.stream()
    unique_classes = set()
    
    for student in all_students:
        student_data = student.to_dict()
        if 'class' in student_data and student_data['class']:
            unique_classes.add(student_data['class'])

    unique_classes = list(unique_classes)  # Convert to list for rendering

    # Apply class filter if POST request
    if request.method == 'POST':
        class_filter = request.form.get('class_filter')
        if class_filter:
            students_ref = students_ref.where('class', '==', class_filter)

    docs = students_ref.stream()
    student_list = [doc.to_dict() for doc in docs]
    
    # Calculate analytics
    total_students = len(student_list)
    avg_tahap_penguasaan_1 = (
        (sum(s['tahap_penguasaan_1'] for s in student_list if 'tahap_penguasaan_1' in s and s['tahap_penguasaan_1'] is not None) / total_students) * 10
    ) if total_students > 0 else 0

    avg_tahap_penguasaan_2 = (
        (sum(s['tahap_penguasaan_2'] for s in student_list if 'tahap_penguasaan_2' in s and s['tahap_penguasaan_2'] is not None) / total_students) * 10
    ) if total_students > 0 else 0

    # Calculate average scores per class
    average_scores_1 = {}
    average_scores_2 = {}
    class_counts = {}
    
    for student in student_list:
        class_name = student.get('class')
        if class_name:
            if class_name not in average_scores_1:
                average_scores_1[class_name] = 0
                class_counts[class_name] = 0
            
            if 'tahap_penguasaan_1' in student and student['tahap_penguasaan_1'] is not None:
                average_scores_1[class_name] += student['tahap_penguasaan_1']
                class_counts[class_name] += 1
            
            if 'tahap_penguasaan_2' in student and student['tahap_penguasaan_2'] is not None:
                if class_name not in average_scores_2:
                    average_scores_2[class_name] = 0
                average_scores_2[class_name] += student['tahap_penguasaan_2']
    
    # Calculate average for each class
    for class_name in average_scores_1:
        average_scores_1[class_name] /= class_counts[class_name] if class_counts[class_name] > 0 else 1
        
    for class_name in average_scores_2:
        average_scores_2[class_name] /= class_counts[class_name] if class_counts[class_name] > 0 else 1
    
    # Get data if tahap_penguasaan_1
    tahap_counts_1 = {i: 0 for i in range(1, 7)}
    for student in student_list:
        tahap_1 = student.get('tahap_penguasaan_1')  # Adjust key according to your data structure
        if tahap_1 is not None:
            if tahap_1 in tahap_counts_1:
                tahap_counts_1[tahap_1] += 1
            else:
                tahap_counts_1[tahap_1] = 1
    
    # Get data if tahap_penguasaan_2
    tahap_counts_2 = {i: 0 for i in range(1, 7)}
    for student in student_list:
        tahap_2 = student.get('tahap_penguasaan_2')  # Adjust key according to your data structure
        if tahap_2 is not None:
            if tahap_2 in tahap_counts_2:
                tahap_counts_2[tahap_2] += 1
            else:
                tahap_counts_2[tahap_2] = 1
    
    return render_template('analytics.html', 
                        total_students=total_students,
                        avg_tahap_penguasaan_1=avg_tahap_penguasaan_1,
                        avg_tahap_penguasaan_2=avg_tahap_penguasaan_2,
                        unique_classes=unique_classes,
                        average_scores_1=average_scores_1,
                        average_scores_2=average_scores_2,
                        tahap_counts_1=tahap_counts_1,
                        tahap_counts_2=tahap_counts_2)  # Pass it to the template

# Route to print analytcis
@app.route('/print_analytics', methods=['POST'])
def print_analytics():
    
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    class_filter = request.form.get('class_filter')
    students_ref = db.collection('students')

    if class_filter:
        students_ref = students_ref.where('class', '==', class_filter)

    docs = students_ref.stream()
    student_list = [doc.to_dict() for doc in docs]

    # Calculate total number of students
    total_students = len(student_list)

    # Calculate average scores for tahap penguasaan 1 and 2
    avg_tahap_penguasaan_1 = (
        (sum(s['tahap_penguasaan_1'] for s in student_list if 'tahap_penguasaan_1' in s and s['tahap_penguasaan_1'] is not None) / total_students) * 10
    ) if total_students > 0 else 0

    avg_tahap_penguasaan_2 = (
        (sum(s['tahap_penguasaan_2'] for s in student_list if 'tahap_penguasaan_2' in s and s['tahap_penguasaan_2'] is not None) / total_students) * 10
    ) if total_students > 0 else 0

    # Calculate average scores per class
    average_scores_1 = {}
    average_scores_2 = {}
    class_counts = {}
    
    for student in student_list:
        class_name = student.get('class')
        if class_name:
            if class_name not in average_scores_1:
                average_scores_1[class_name] = 0
                class_counts[class_name] = 0
            
            if 'tahap_penguasaan_1' in student and student['tahap_penguasaan_1'] is not None:
                average_scores_1[class_name] += student['tahap_penguasaan_1']
                class_counts[class_name] += 1
            
            if 'tahap_penguasaan_2' in student and student['tahap_penguasaan_2'] is not None:
                if class_name not in average_scores_2:
                    average_scores_2[class_name] = 0
                average_scores_2[class_name] += student['tahap_penguasaan_2']
    
    # Calculate average for each class
    for class_name in average_scores_1:
        average_scores_1[class_name] /= class_counts[class_name] if class_counts[class_name] > 0 else 1
        
    for class_name in average_scores_2:
        average_scores_2[class_name] /= class_counts[class_name] if class_counts[class_name] > 0 else 1
    
    # Calculate the distribution of tahap_penguasaan 1 and 2
    tahap_counts_1 = {i: 0 for i in range(1, 7)}
    tahap_counts_2 = {i: 0 for i in range(1, 7)}

    for student in student_list:
        if student.get('tahap_penguasaan_1') is not None:
            tahap_counts_1[student['tahap_penguasaan_1']] += 1
        if student.get('tahap_penguasaan_2') is not None:
            tahap_counts_2[student['tahap_penguasaan_2']] += 1

    # Generate a grouped bar plot for tahap_penguasaan distribution
    plt.clf()  # Clear the previous plot

    bar_width = 0.35  # Width of the bars
    index = np.arange(len(tahap_counts_1.keys()))  # X locations for the groups

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Creating the bars
    bars1 = ax.bar(index, tahap_counts_1.values(), bar_width, label='Tahap Penguasaan 1', alpha=0.7)
    bars2 = ax.bar(index + bar_width, tahap_counts_2.values(), bar_width, label='Tahap Penguasaan 2', alpha=0.7)

    # Adding labels and title
    ax.set_xlabel('Tahap Penguasaan')
    ax.set_ylabel('Number of Students')
    ax.set_title(f'Tahap Penguasaan Distribution for Class {class_filter}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tahap_counts_1.keys())
    ax.legend()

    # Adding value labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    # Save the plot to a PNG image and convert it to base64 for embedding in HTML
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Prepare student details for the detailed scores table
    student_details = [
        {
            'name': student.get('name'),
            'class': student.get('class'),
            'tahap_penguasaan_1': student.get('tahap_penguasaan_1'),
            'tahap_penguasaan_2': student.get('tahap_penguasaan_2')
        }
        for student in student_list
    ]

    # Render the print_analytics template with the calculated data
    return render_template(
        'print_analytics.html',
        class_name=class_filter,
        total_students=total_students,
        avg_tahap_penguasaan_1=avg_tahap_penguasaan_1,
        avg_tahap_penguasaan_2=avg_tahap_penguasaan_2,
        average_scores_1=average_scores_1,
        average_scores_2=average_scores_2,
        plot_url=plot_url,
        student_details=student_details
    )

# Route to predict marks using regression
@app.route('/marks-prediction', methods=['GET', 'POST'])
def marks_prediction():
    if not is_logged_in():  # Check if user is logged in
        flash('You must be logged in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    unique_classes = {doc.to_dict()['class'] for doc in db.collection('students').stream()}  # Fetch unique classes
    predictions = []  # Initialize predictions list
    last_update_time = None  # Initialize last update time
    selected_class = None  # Initialize selected_class with None to avoid UnboundLocalError

    if request.method == 'POST':
        selected_class = request.form.get('selected_class')  # Retrieve selected class from form
        print(f"Selected class: {selected_class}")  # Debugging line

        students_ref = db.collection('students')

        # Filter students by selected class if provided
        if selected_class:
            students_ref = students_ref.where('class', '==', selected_class)

        docs = students_ref.stream()
        student_list = [{**doc.to_dict(), 'id': doc.id} for doc in docs]
        print(f"Student list: {student_list}")  # Debugging line
        
        # Capture the current time as the last update time
        last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check which button was clicked
        if 'show_predictions' in request.form:
            # Show prediction marks button clicked
            for student in student_list:
                # Collect the necessary information
                if student.get('tahap_penguasaan_1') is not None and student.get('tahap_penguasaan_2') is not None:
                    # Append data to predictions
                    predictions.append({
                        'name': student['name'],  # Assuming 'name' is a field in your student document
                        'class': student['class'],
                        'tahap_penguasaan_1': student['tahap_penguasaan_1'],
                        'tahap_penguasaan_2': student['tahap_penguasaan_2'],
                        'predicted_tahap_penguasaan': student.get('predicted_tahap_penguasaan', None)  # Get predicted value if it exists
                    })

        else:
            # Predict button clicked
            X = []
            y = []
            
            for student in student_list:
                print(f"Student data: {student}")  # Debugging line
                if student.get('tahap_penguasaan_1') is not None and student.get('tahap_penguasaan_2') is not None:
                    X.append([student['tahap_penguasaan_1'], student['tahap_penguasaan_2']])
                    y.append(student['tahap_penguasaan_2'])  # Using tahap_penguasaan_2 as the target

            if X and y:  # Ensure we have data for regression
                model = LinearRegression()
                model.fit(X, y)
                predicted_scores = model.predict(X)

                # Save predictions back into Firestore and display them
                for i, student in enumerate(student_list):
                    predicted_value = predicted_scores[i]
                    db.collection('students').document(student['id']).update({
                        'predicted_tahap_penguasaan': predicted_value
                    })

                    predictions.append({
                        'name': student['name'],
                        'class': student['class'],
                        'tahap_penguasaan_1': student['tahap_penguasaan_1'],
                        'tahap_penguasaan_2': student['tahap_penguasaan_2'],
                        'predicted_tahap_penguasaan': predicted_value
                    })

    # Pass selected_class and last_update_time even in GET request when no form submission has been made
    return render_template('marks_prediction.html', 
                           predictions=predictions, 
                           unique_classes=unique_classes, 
                           selected_class=selected_class,
                           last_update_time=last_update_time)

# Set up logging
logging.basicConfig(level=logging.INFO)



# Run the app
if __name__ == '__main__':
    # Get the port from the environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'false').lower() in ['true', '1']) #for hosting app
