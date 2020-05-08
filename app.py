from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

def load_model():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

@app.route('/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        model = load_model()
        prediction = []

        meanRadius = float(request.form.get('meanRadius'))
        prediction.append(meanRadius)

        meanTexture = float(request.form.get('meanTexture'))
        prediction.append(meanTexture)

        meanPerimeter = float(request.form.get('meanPerimeter'))
        prediction.append(meanPerimeter)

        meanArea = float(request.form.get('meanArea'))
        prediction.append(meanArea)

        meanSmoothness = float(request.form.get('meanSmoothness'))
        prediction.append(meanSmoothness)

        meanCompactness = float(request.form.get('meanCompactness'))
        prediction.append(meanCompactness)

        meanConcavity = float(request.form.get('meanConcavity'))
        prediction.append(meanConcavity)

        meanConcavePoints = float(request.form.get('meanConcavePoints'))
        prediction.append(meanConcavePoints)

        meanSymmetry = float(request.form.get('meanSymmetry'))
        prediction.append(meanSymmetry)

        meanFractalDimension = float(request.form.get('meanFractalDimension'))
        prediction.append(meanFractalDimension)

        prediction_data = []
        prediction_data.append(prediction)
        
        pred = model.predict(prediction_data)

        if pred[0] == 0:
            result = 'Malignant'
        else:
            result = 'Benign'
        
        return render_template('result.html', result = result)
    else:
        return render_template('index.html')