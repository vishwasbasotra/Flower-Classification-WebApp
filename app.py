from flask import Flask,render_template,session,url_for, redirect
import os
from flask_wtf import FlaskForm
import numpy as np
from wtforms import TextField, SubmitField, StringField, validators
from wtforms.validators import NumberRange, Required
import tensorflow as tf
import joblib

def return_prediction(model,scaler,sample_json):

    # for larger data features, you should write a for loop
    # that builds out this array for you 

    s_len = sample_json["sepal_length"]
    s_width = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_width = sample_json["petal_length"]
    
    flower = [[s_len,s_width,p_len,p_width]]
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    flower = scaler.transform(flower)
    
    class_ind = model.predict_classes(flower)[0]
    
    return classes[class_ind]

flower_model = tf.keras.models.load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

images = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = images
app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

class FlowerForm(FlaskForm):

    s_len = StringField("Sepal Length", validators=[Required()], render_kw={"placeholder": "Enter Between : 4 - 8"})
    s_width = StringField("Sepal Width",[validators.required()], render_kw={"placeholder": "Enter Between : 2 - 4.4"})
    p_len = StringField("Petal Length",[validators.required()], render_kw={"placeholder": "Enter Between : 1 - 7"})
    p_width = StringField("Petal Length",[validators.required()], render_kw={"placeholder": "Enter Between : 0 - 2.5"})

    submit = SubmitField("Predict")

@app.route("/",methods=['GET','POST'])
def index():
    
    form = FlowerForm()
    favicon = os.path.join(app.config['UPLOAD_FOLDER'], 'favicon.png')

    if form.validate_on_submit():
        session['s_len'] = form.s_len.data
        session['s_width'] = form.s_width.data
        session['p_len'] = form.p_len.data
        session['p_width'] = form.p_width.data

        return redirect(url_for("prediction"))
    return render_template('index.html', form=form,favicon=favicon)

@app.route('/prediction')
def prediction():
    content = {}
    flower = os.path.join(app.config['UPLOAD_FOLDER'], 'iris.jpg')
    favicon = os.path.join(app.config['UPLOAD_FOLDER'], 'favicon.png')

    content['sepal_length'] = float(session['s_len'])
    content['sepal_width'] = float(session['s_width'])
    content['petal_length'] = float(session['p_len'])
    content['petal_width'] = float(session['p_width'])

    results = return_prediction(flower_model,flower_scaler,content)
    return render_template('prediction.html',results=results,  user_image = flower, favicon=favicon)

if __name__ == '__main__':
    app.run(debug=True)