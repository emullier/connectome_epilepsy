from flask import Flask, render_template, Response, request, jsonify
import subprocess
import os
import json
from flask import Flask, render_template, Response, jsonify
import subprocess
import os
import traceback


# Create a Flask app instance with custom folders
app = Flask(__name__, template_folder='public', static_folder='public/static')
@app.route('/')
def index():
    return render_template('index.html')
# Path to the configuration file
CONFIG_FILE_PATH = 'config.json'

# Load the configuration from the file
def load_config():
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as file:
            return json.load(file)
    return {}

# Save the configuration to the file
def save_config(config):
    with open(CONFIG_FILE_PATH, 'w') as file:
        json.dump(config, file, indent=4)

@app.route('/get-config', methods=['GET'])
def get_config():
    config = load_config()
    return jsonify(config)

@app.route('/save-config', methods=['POST'])
def save_config_route():
    try:
        config = request.get_json()
        if not isinstance(config, dict):
            return jsonify({"error": "Invalid data format"}), 400
        save_config(config)
        return jsonify({"message": "Configuration saved successfully"}), 200
    except Exception as e:
        traceback.print_exc()  # This will print the full error stack trace to the console
        return jsonify({"error": str(e)}), 500

def generate_console_output():
    process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in iter(process.stdout.readline, ''):
        yield line + '<br/>'
    process.stdout.close()
    process.wait()

#@app.route('/run-script')
#def run_script():
#    return Response(generate_console_output(), mimetype='text/html')

@app.route('/run-script')
def run_script():
    def generate():
        process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''):
            yield line.decode('utf-8')
        for line in iter(process.stderr.readline, b''):
            yield line.decode('utf-8')
        process.stdout.close()
        process.wait()

    return Response(generate(), mimetype='text/plain')

def load_seen_files(section):
    try:
        with open(f'seen_files_{section}.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_seen_files(section, files):
    with open(f'seen_files_{section}.json', 'w') as file:
        json.dump(list(files), file)
        
@app.route('/get-new-figures', methods=['GET'])
def get_new_figures():
    section = request.args.get('section')
    figures_dir = f'public/static/images/{section}/'  # Assuming images are organized by section
    
    if not os.path.exists(figures_dir):
        return jsonify({"new_files": []})

    existing_files = os.listdir(figures_dir)
    
    # Load previously seen files for the section (implement `load_seen_files` accordingly)
    seen_files = set(load_seen_files(section))  
    new_files = list(set(existing_files) - seen_files)

    # Update seen files list for the section
    save_seen_files(section, existing_files)  # Implement `save_seen_files` to save the updated list

    return jsonify({"new_files": new_files})

if __name__ == '__main__':
    app.run(debug=True, port=3000)



#### Future developments
## Plot interactively with Flask and Bokeh tutorial : https://medium.com/@andresberejnoi/interactive-plots-with-bokeh-and-flask-andres-berejnoi-64fbfd328cce