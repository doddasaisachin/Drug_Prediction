from flask import Flask,render_template,request
import pickle

application=Flask(__name__)
app=application

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        age=request.form.get('age')
        sex=request.form.get('sex')
        bp=request.form.get('bp')
        cholestrol=request.form.get('cholestrol')
        na_to_k=request.form.get('na_to_k')

        sex_enc=pickle.load(open('sex_encoder.pkl','rb'))
        bp_enc=pickle.load(open('bp_encoder.pkl','rb'))
        chol_enc=pickle.load(open('cholestrol_encoder.pkl','rb'))

        model=pickle.load(open('model.pkl','rb'))
        new_sex=sex_enc.transform([sex])[0]
        new_bp=bp_enc.transform([bp])[0]
        new_chol=chol_enc.transform([cholestrol])[0]

        predicted_drup=model.predict([[age,new_sex,new_bp,new_chol,na_to_k]])
        predicted_drup=predicted_drup[0]

        return render_template('index.html',result=predicted_drup)

if __name__=='__main__':
    app.run('0.0.0.0')