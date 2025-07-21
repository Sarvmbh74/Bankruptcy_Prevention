import numpy as np
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt  # ✅ Added for the chart

from PIL import Image

# Load the trained model
pickle_in = open("model_poly.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome ALL"

def predict_bankruptcy(industrial_risk, management_risk, financial_flexibility,
                       credibility, competitiveness, operating_risk):
    prediction = classifier.predict([[industrial_risk, management_risk, financial_flexibility,
                                      credibility, competitiveness, operating_risk]])
    return prediction[0]

def main():
    st.title("Bankruptcy Detector")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bankruptcy Detector ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("Select risk levels for each factor (0 = Low, 0.5 = Medium, 1 = High)")

    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, step=0.5)
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, step=0.5)
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, step=0.5)
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5, step=0.5)
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5, step=0.5)
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, step=0.5)

    result = ""
    if st.button("Predict"):
        output = predict_bankruptcy(industrial_risk, management_risk, financial_flexibility,
                                    credibility, competitiveness, operating_risk)
        if output == 0:
            result = "Risk is Low"
        elif output == 0.5:
            result = "Risk is Medium"
        elif output == 1:
            result = "Risk is High"
        else:
            result = f"Risk: {output}"

        st.success(f'The output is: {result}')

        # ✅ Show a bar chart of selected risks
        risk_labels = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                       'Credibility', 'Competitiveness', 'Operating Risk']
        risk_values = [industrial_risk, management_risk, financial_flexibility,
                       credibility, competitiveness, operating_risk]

        st.subheader("Your Risk Profile")
        fig, ax = plt.subplots()
        ax.bar(risk_labels, risk_values, color='tomato')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Risk Level')
        ax.set_title('Selected Risk Factors')
        plt.xticks(rotation=30)
        st.pyplot(fig)

    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()
