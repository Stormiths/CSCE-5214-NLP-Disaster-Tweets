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
    model = request.form['model']
    pythonExe = sys.executable
    if model == 'logistic_regression':
        # Run the LR script and use the output
        process = subprocess.Popen(
            [pythonExe, 'logistic_regression.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            output = f"An error occurred:\n{stderr}"
        else:
            output = stdout

        # Place the output on the new page
        html = f'''
        <html>
        <head>
            <title>Logistic Regression Output</title>
        </head>
        <body>
            <h1>Logistic Regression Model Output</h1>
            <pre>{output}</pre>
            <a href="/">Go Back</a>
        </body>
        </html>
        '''
        return render_template_string(html)
    elif model == 'bert':
        # To easily deploy, follow the same format as the above if statement
        # make sure to use the python_exe to run the script, otherwise
        # there is problems with libraries.
        html = '''
        <html>
        <head>
            <title>BERT Output</title>
        </head>
        <body>
            <h1>BERT Model Output</h1>
            <p>BERT model functionality is not implemented yet.</p>
            <a href="/">Go Back</a>
        </body>
        </html>
        '''
        return render_template_string(html)
    else:
        return 'Invalid option.'

if __name__ == '__main__':
    app.run(debug=True)
