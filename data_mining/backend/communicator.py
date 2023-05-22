from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        print(file.filename)
        if file.filename == '':
            result = {'success': False, 'message': 'No file selected.'}
            return jsonify(result)
        if file:
            try:       
                line = file.readline()          
                while line:  
                    print (line),                 
                    line = file.readline()  
                file.close()  
                result = {'success': True, 'name': 'star', 'score' : '0.75' }
                return jsonify(result)
            except:
                result = {'success': False, 'message': 'Failed to parse file.'}
                return jsonify(result)
    else:
        result = {'success': False, 'message': 'Failed to parse file.'}
        return jsonify(result)

if __name__ == "__main__":
    print('run 0.0.0.0:12225')
    app.run(host='0.0.0.0', port=12225)

