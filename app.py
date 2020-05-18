from flask import Flask,request, url_for, redirect, render_template
import pickle
import joblib
import numpy as np
app = Flask(__name__)

#model=pickle.load(open('mymodel1.pkl','rb'))
model = joblib.load('mymodel.pkl','r')

@app.route('/')
def hello_world():
    return render_template("myform.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    extra=[-0.71153357, -0.70783419, -0.74697396, -0.72922197, -0.74699153, -0.63174862, -0.67584013, -0.57187389, -0.66951814, -0.74130504, -0.57179849, -0.84596189, 0.0928208, 0.1291473, 0.14592517, -0.1753366, -0.00843346, -0.14715714, -0.32890122]
    final=[np.array(int_features+extra)]
    print(int_features)
    print(final)
    prediction=model.predict([final])
    output=prediction[0]
    output=1

    if output==1:
        return render_template('myform.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('myform.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)