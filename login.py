import streamlit as st
import firebase_admin

from firebase_admin import credentials
from firebase_admin import auth

import pandas as pd
import bcrypt

if not firebase_admin._apps:
    cred = credentials.Certificate('hnd-project-25934-b3131f2827b7.json')
    firebase_admin.initialize_app(cred)


def login():

    def f():

        if not email or not password:
            st.warning("All fields are required.")
            return

        if len(password) < 6:
            st.error("Password must be at least 6 characters long.")
            return


        try:
            user = auth.get_user_by_email(email)
            # print(user.uid)
            st.session_state.logged_in = True
            st.session_state.username = user.uid
            st.success(user.uid + " logged in")
            st.balloons()
            st.session_state.page = "Dashboard"

        except:
            st.warning('Login Failed')

    st.subheader("Login to Your Account")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")
    
    st.button("Login",on_click=f)
    # if st.button("Login"):
        # try:
        #     users = pd.read_csv("user_data.csv")
        # except FileNotFoundError:
        #     st.error("No user database found. Please sign up first.")
        #     return

        # user = users[users["username"] == username]

        # if not user.empty and bcrypt.checkpw(password.encode(), user.iloc[0]["password"].encode()):
        #     st.session_state.logged_in = True
        #     st.session_state.username = username
        #     st.session_state.page = "Dashboard"  # 👈 Trigger redirect
        #     st.success("Login successful!")
        # else:
        #     st.error("Invalid username or password.")
