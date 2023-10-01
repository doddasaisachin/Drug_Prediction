import pandas as pd
import numpy as np

df=pd.read_csv('data/drug.csv')

from sklearn.preprocessing import LabelEncoder

sex_enc=LabelEncoder()

drug_df=pd.DataFrame(df.iloc[:,0])

drug_df['enc_sex']=sex_enc.fit_transform(df['Sex'])



bp_enc=LabelEncoder()

drug_df['enc_bp']=bp_enc.fit_transform(df['BP'])



df['Cholesterol'].value_counts()

enc_cholestrol=LabelEncoder()

drug_df['enc_cholestrol']=enc_cholestrol.fit_transform(df['Cholesterol'])



drug_df['na_to_k']=df['Na_to_K']



from imblearn.over_sampling import SMOTE

smote=SMOTE()

new_x,new_y=smote.fit_resample(X=drug_df,y=df['Drug'])



import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(new_x,new_y,test_size=0.15)

xtrain.shape,ytrain.shape,xtest.shape,ytest.shape

from sklearn.ensemble import RandomForestClassifier

rfc_model=RandomForestClassifier()

rfc_model.fit(xtrain,ytrain)



import pickle

pickle.dump(sex_enc,open('sex_encoder.pkl','wb'))

pickle.dump(bp_enc,open('bp_encoder.pkl','wb'))

pickle.dump(enc_cholestrol,open('cholestrol_encoder.pkl','wb'))

pickle.dump(rfc_model,open('model.pkl','wb'))

