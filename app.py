from flask import Flask, render_template, request
import joblib  
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load model
model = joblib.load('KNNClassifier.pkl')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        data_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model.predict(data_array)

        classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = classes[int(prediction[0])]

        return render_template('index.html', prediction_text=f'Kết quả: {predicted_class}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
