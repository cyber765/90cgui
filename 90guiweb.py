

import streamlit as st
import numpy as np
import torch # For building the networks
import torchtuples as tt # Some useful functions
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import pandas as pd
import matplotlib.pyplot as plt
import pycox
import os




#os.chmod("/home/xffktz/pyproject/90cguiweb",0o777)
tabs_font_css = """
<style>
div[class*="stNumberInput"] label p {
  font-size: 30px;
}
div[class*="row-widget stButton"] button p {
  font-size: 30px;
}
</style>
"""


st.write(tabs_font_css, unsafe_allow_html=True)

st.title('Deephit model')


PGA = st.number_input("PGA")
PRO = st.number_input('PRO')


ALA = st.number_input('ALA')

GLY = st.number_input('GLY')


HIS = st.number_input('HIS')

day = st.number_input('Predict Day',min_value=0,max_value=80)

model = torch.load("deephit5sig.pt", weights_only=False)

pdata1 = pd.DataFrame({"dat": (PGA, PRO, ALA, GLY, HIS)})
pdata2 = pdata1.transpose().values.astype("float32")

pred1 = model.interpolate(1000).predict_surv_df(pdata2)
ev1 = EvalSurv(pred1,
               np.array(1),
               np.array(1))
s = ev1.surv_at_times(day)
spro = s*100


if st.button('Predict') :
    st.write(f"<p style='font-size: 30px;'>Surv at {day}-day : %0.2f%%</p>"%spro, unsafe_allow_html=True)
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca
    plt.plot(pred1.index,pred1[0])
    plt.xlabel('Day')
    plt.ylabel('Surv')
    plt.ylim(-0.1,1.1)
    st.pyplot(fig)



