import streamlit as st
import pymunk
import numpy as np
import scipy
import matplotlib.pyplot as plt
import anthropic

st.title("Environment Setup Verification")
st.write("If you see this, Streamlit is running successfully!")

# Quick check on packages
st.success(f"Pymunk version: {pymunk.version}")

st.success(f"Numpy version: {np.__version__}")
st.success(f"Scipy version: {scipy.__version__}")
