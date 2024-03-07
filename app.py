from flask import Flask, jsonify, render_template, request
import model  # Import your script

app = Flask(__name__, static_url_path='/static', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        precision = model.main()  # Call your main function
        return jsonify({'precision': precision})

if __name__ == '__main__':
    app.run(debug=True)