from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app

def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol):
    with open("finalized_model_Decision_Tree.pickle", 'rb') as f:
        model = pickle.load(f)
    with open("pca_model.sav", 'rb') as f:
        pca_model = pickle.load(f)
    predict = model.predict(pca_model.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,density,pH,sulphates,alcohol]]))
    if predict[0] == 3:
        result = 'Bad'
    elif predict[0] == 4 :
        result = 'Below Average'
    elif predict[0]==5:
        result = 'Average'
    elif predict[0] == 6:
        result = 'Good'
    elif predict[0] == 7:
        result = 'Very Good'
    else :
        result = 'Excellent'
    return result

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity=float(request.form['fixed acidity'])
            volatile_acidity = float(request.form['volatile acidity'])
            citric_acid = float(request.form['citric acid'])
            residual_sugar = float(request.form['residual sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free sulfur dioxide'])
            total_sulfur_dioxide = float(request.form['total sulfur dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
            prediction = predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction = prediction)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')

# testing via postman

@app.route('/from_postman',methods=['POST']) # route to show the predictions in a web UI
@cross_origin()
def from_postman():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity = float(request.json['fixed acidity'])
            volatile_acidity = float(request.json['volatile acidity'])
            citric_acid = float(request.json['citric acid'])
            residual_sugar = float(request.json['residual sugar'])
            chlorides = float(request.json['chlorides'])
            free_sulfur_dioxide = float(request.json['free sulfur dioxide'])
            total_sulfur_dioxide = float(request.json['total sulfur dioxide'])
            density = float(request.json['density'])
            pH = float(request.json['pH'])
            sulphates = float(request.json['sulphates'])
            alcohol = float(request.json['alcohol'])
            prediction = predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return jsonify({'prediction':prediction})
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app