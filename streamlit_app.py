# from email.policy import default
# from tkinter import CENTER
# import Tkinter as tk
import streamlit as st 
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

import codecs
# import pickle
import joblib
import requests

#----------------------------------------------#

import json
import pandas as pd
import numpy as np
# import seaborn as sn
# import matplotlib.pyplot as plt
# from statistics import mean, stdev

#----------------------------------------------#
# import sklearn
# from sklearn import preprocessing
# from sklearn.model_selection import StratifiedKFold
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification
# from sklearn.metrics import f1_score,accuracy_score
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import balanced_accuracy_score
# from imblearn.ensemble import BalancedRandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
# from sklearn.preprocessing import StandardScaler

#------------------------------------------------#

from rdkit.Chem import Descriptors, Lipinski, rdqueries
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem, DataStructs
# from rdkit.Chem.Draw import SimilarityMaps, IPythonConsole
# from rdkit.Chem.Draw import rdMolDraw2D
from chembl_webresource_client.new_client import new_client
# from pikachu.general import draw_smiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# import os
# from os import path
# import zipfile
# import glob
# import random

# from keras.utils import np_utils

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras import layers
# from tensorflow.keras import initializers
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

st.set_page_config(layout="wide")

#### tab bar ####
selected = option_menu(
    menu_title=None, 
    options=["Home", "About us", "Predict new SMILES molecule","Check your SMILES molecule"], 
    icons=["house","book","check2-all","search"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal", #แนวนอน
    styles={
        "container": {"padding": "0!important", "background-color": "#CAF1E9"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"8px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#B5E6E5"},
    }
)

#### sticker image ####
def load_lottiefile(filepath: str):
    with open (filepath,"r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#### import html ####
import streamlit.components.v1 as stc 
def st_webpage(page_html,width=1370,height=2000):
    page_file = codecs.open(page_html,'r')
    page =page_file.read()
    stc.html(page,width=width, height=height , scrolling = False)

#### selected tab bar ####
if selected =="Home":
    st.title(f"Welcome to Developing web applications for Breast Cancer Novel Drug Discovery Using the ChemBL Database and Deep Learning approach.")
    thai_title = '<p style="ont-family:Courier; color:#3D0E04; font-size: 18px;"> การพัฒนาเว็บแอพพลิเคชั่นสำหรับการค้นคว้ายาใหม่สำหรับมะเร็งเต้านมด้วยฐานข้อมูล ChemBL และแนวทางการเรียนรู้เชิงลึก</p>'
    st.markdown(thai_title, unsafe_allow_html=True)
    
    # ---- LOAD ASSETS ----
    st.write("##")
    lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_nw19osms.json")
    st_lottie(lottie_coding, height=350, key="coding")
    # index_title = '<p style="ont-family:Courier; color:Black; font-size: 20px;">This is the introduction of a drug molecule from an old SMILES molecule into a new one from the eight target protein targets mTOR, HER2, aromatase, CDK4/6, Trop-2, Estrogen Receptor, PI3K and Akt of breast cancer. To researchers or individuals who wish to discover drugs or produce drugs in the drug discovery process to explore the possibilities of molecules before studying further research into the production of future drugs.</p>'
    # st.markdown(index_title, unsafe_allow_html=True)
    st.info('This is the introduction of a drug molecule from an old SMILES molecule into a new one from the eight target protein targets mTOR, HER2, aromatase, CDK4/6, Trop-2, Estrogen Receptor, PI3K and Akt of breast cancer. To researchers or individuals who wish to discover drugs or produce drugs in the drug discovery process to explore the possibilities of molecules before studying further research into the production of future drugs.')
   

if selected =="About us":
    with st.container():
        st.title("About us 👥")
        Welcome_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; ">  Welcome to design drug discovery of breast cancer from pre-clinical stage and check of new SMILES molecules.</p>'
        st.markdown(Welcome_title, unsafe_allow_html=True)
        st.write("[Database >](https://www.ebi.ac.uk/chembl/)")
            
    with st.container():
        st.write("---")
        st.header("Goal of the projects")
        st.write("""" The goal of this project is to introduce non-toxic drug molecules at the pre-clinical stage before the results of the study can be used to produce or create future drugs. " """)
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("##")
            Ideal_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; "> 🧬 Ideal.</p>'
            st.markdown(Ideal_title, unsafe_allow_html=True)
            st.info(
                """ 
                - It is a process of drug discovery that can Commonly discovered by predicting protein target molecules on web applications
                  And the drug molecules obtained from the prediction can be further studied before producing or developing drugs in the future.
             """)
            st.write("##")
            Reality_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size:20px; "> 🧬 Reality.</p>'
            st.markdown(Reality_title, unsafe_allow_html=True)
            st.info(
                """ 
                - In reality, the field of medicine is more complex than we think, starting with observing, experimenting and researching the properties of the natural surroundings. 
                  development of medicinal substances with the synthesis of chemical compounds or compounds that imitate important substances in nature, which the discovery and manufacture of each drug knowledge 
                  required in many disciplines Important substances that have medicinal properties and are available for sale. must be extracted synthesis or compound analysis number of more than ten thousand species 
                  To be selected to study the potency and toxicity of the drug in vitro.
            """)

            st.write("##")
            Consequences_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; "> 🧬 Consequences.</p>'
            st.markdown(Consequences_title, unsafe_allow_html=True)
            st.info(
                """ 
                - To discover and produce the desired drug If there is no application or technology to help at all It takes an average period of up to 15 years and costs a minimum of approximately $800 million. 
                  Therefore, each discovery of a drug takes a long time and a large budget. The group therefore chose to discover new drugs. together with a machine learning model to help mitigate this problem for future drug development circles.
             """)
            st.write("##")
        with right_column:
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            lottie2_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_pk5mpw6j.json")
            st_lottie(lottie2_coding, height=400,  key="coding")  


if selected =="Predict new SMILES molecule":
    Welcome_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; ">  Welcome to design drug discovery of breast cancer from pre-clinical stage and check of new smiles molecules.</p>'
    st.markdown(Welcome_title, unsafe_allow_html=True)


if selected =="Check your SMILES molecule":
    st.title(f"Check your SMILES molecule")
    st.write(""" SMILES = Simplified Molecular Input Line Entry Specification """)
    st.write("The Lipinski's Rule stated the following: Molecular weight < 500 Dalton, Octanol-water partition coefficient (LogP) < 5, Hydrogen bond donors < 5, Hydrogen bond acceptors < 10 ")
    st.write("Atoms are shown by atomic symbols")
    st.write("Hydrogen atoms are assumed to fill spare valencies")
    st.write("Adjacent atoms are connected by single bonds")
    st.write("""double bonds are shown by "=" """)
    st.write("""triple bonds are shown by "#" """)
    st.write("Branching is indicated by parenthesis")
    st.write("Ring closures are shown by pairs of matching digits")
    
    canonical_smiles = st.text_input("1.Enter your SMILES molecules string")

    if st.button("Predict"):
        def draw_compound(canonical_smiles):
                    mpicmole = Chem.MolFromSmiles(canonical_smiles)
                    weight = Descriptors.MolWt(mpicmole)
                    return Draw.MolsToGridImage(mpicmole, size=(500,500))
        col1, col2 = st.columns(2)
        col1.write('')
        col1.write("""<style>.font-family {font-size:15px !important;}</style>""", unsafe_allow_html=True)
        col1.write('<p class="font-family">This is your smile molecule image</p>', unsafe_allow_html=True)
        col1.image(draw_compound(canonical_smiles))

        st.write(f"Test")

