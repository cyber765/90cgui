








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

os.chmod("/home/xffktz/pyproject/90cguiweb",0o777)


st.title('Deephit model')


PGA = st.number_input('PGA')
PRO = st.number_input('PRO')


ALA = st.number_input('ALA')

GLY = st.number_input('GLY')


HIS = st.number_input('HIS')


model = torch.load("deephit5sig.pt")

pdata1 = pd.DataFrame({"dat": (PGA, PRO, ALA, GLY, HIS)})
pdata2 = pdata1.transpose().values.astype("float32")

pred1 = model.interpolate(10).predict_surv_df(pdata2)
ev1 = EvalSurv(pred1,
               np.array(1),
               np.array(1))
s30 = ev1.surv_at_times(30)
s30pro = s30[0]*100
s60 = ev1.surv_at_times(60)
s60pro = s60[0]*100



if st.button('Predict') :
    st.write('Surv at 30-day : %0.2f%%' % s30pro)
    st.write('Surv at 60-day : %0.2f%%' % s60pro)
    fig = plt.figure(figsize=(8,6))
    plt.plot(pred1.index,pred1[0])
    plt.xlabel('Day')
    plt.ylabel('Surv')
    plt.ylim(-0.1,1.1)
    st.pyplot(fig)



