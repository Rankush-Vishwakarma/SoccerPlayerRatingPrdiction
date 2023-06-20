from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle
standard_scaler = StandardScaler()
with open(r"finalized_SVR.sav", "rb") as input_file:
    model = pickle.load(input_file)
app = Flask(__name__)

labels = ['height',
 'weight',
 'potential',
 'crossing',
 'finishing',
 'heading accuracy',
 'short passing',
 'volleys',
 'dribbling',
 'curve',
 'free kick accuracy']

@app.route('/')
def index():
    return render_template('index.html', labels=labels)

@app.route('/predict', methods=['POST'])
def predict():
    inputs = {}
    for label in labels:
        value = request.form.get(label)
        inputs[label] = float(value) if value else 0.0

    # Perform your rating prediction based on the inputs
    # Replace the following line with your prediction code
    values = list(inputs.values())
    scaled_values = standard_scaler.fit_transform([values])
    print(type(scaled_values))
    prediction = round(model.predict(scaled_values)[0],3)   

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
