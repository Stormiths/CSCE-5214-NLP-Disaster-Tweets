from flask import Flask, render_template_string, request
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def home():
    html = '''
    <html>
    <head>
        <title>CSCE 5214 Group 3 Project 1</title>
    </head>
    <body>
        <h1>CSCE 5214 Group 3 Project 1</h1>
        <form action="/run_model" method="post">
            <button name="model" value="logistic_regression" type="submit">Run Logistic Regression</button>
            <button name="model" value="bert" type="submit">Run BERT</button>
        </form>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/run_model', methods=['POST'])
def run_model():
    python_exe = sys.executable
    model = request.form['model']
    if model == 'logistic_regression':
        # Run the logistic regression script
        subprocess.call([python_exe, 'logistical_regression.py'])
        return 'Logistic Regression model has been run.'
    elif model == 'bert':
        # Replace when bert method is given
        return 'BERT model functionality is not implemented yet.'
    else:
        return 'Invalid option.'

if __name__ == '__main__':
    app.run(debug=True)
