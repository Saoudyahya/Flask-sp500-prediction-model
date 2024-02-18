from flask import Flask, jsonify
import model  # Import your script
app = Flask(__name__)

@app.route('/')
def predict():
    precision = model.main()  # Call your main function
    return jsonify({'precision': precision})

if __name__ == '__main__':
    app.run(debug=True)