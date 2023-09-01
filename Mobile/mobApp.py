from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Mobilemodel.pkl', 'rb'))
scaler = pickle.load(open('StandardScaler.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("front.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    for key in request.form:
        value = request.form[key]
        if value:
            features.append(float(value))

    final = np.array([features])
    new_data_scaled = scaler.transform(final)
    output = model.predict(new_data_scaled)
    if output == 0:
        output_label = 'very low'
    elif output == 1:
        output_label = 'low'
    elif output == 2:
        output_label = 'medium'
    elif output == 3:
        output_label = 'high'
    else:
        output_label = 'unknown'
    
    return render_template('front.html', pred='Price level is: {}'.format(output_label))

if __name__ == '__main__':
    app.run(debug=True)
