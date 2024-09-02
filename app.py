import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


@st.cache_data
def load_data():

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df,target=load_data()

model= RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])

st.sidebar.title('Input Features')

sepal_length= st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width= st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length= st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width= st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data= [[sepal_length,sepal_width,petal_length,petal_width]]

#Prediction

prediction = model.predict(input_data)
predicted_species = target[prediction[0]]

st.write('Prediction')
st.write(f"The predicted species is: {predicted_species}")
st.line_chart([sepal_length,sepal_width,petal_length,petal_width])

# Visualization: Scatter plot of the input data
fig, ax = plt.subplots()

# Plot the entire dataset
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='deep', ax=ax)

# Highlight the input data point
ax.scatter(sepal_length, sepal_width, color='red', s=100, label="Predicted Species")
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)