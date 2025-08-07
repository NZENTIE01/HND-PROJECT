import streamlit as st
import os
import pandas as pd
import bcrypt

import firebase_admin

from firebase_admin import credentials, auth

if not firebase_admin._apps:
    cred = credentials.Certificate('hnd-project-25934-b3131f2827b7.json')
    firebase_admin.initialize_app(cred)


def signup():
    st.subheader("Create a New Account")
    username = st.text_input("Username")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):

        if not username or not email or not password:
            st.warning("All fields are required.")
            return

        if len(password) < 6:
            st.error("Password must be at least 6 characters long.")
            return

        if username and email and password:
            hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        try:

            user = auth.create_user(email = email, password = hashed_pw, uid = username)

            st.success('Account created Successfully')
            st.markdown('Please Login using your email and password')
            st.balloons()

        except auth.EmailAlreadyExistsError:
            st.error("This email is already in use. Try logging in or use another email.")
        except auth.InvalidPasswordError:
            st.error("The password provided is invalid or too short.")
        # if password != confirm_password:
        #     st.error("Passwords do not match.")
        #     return

        # hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # # Save user
        # if not os.path.exists("user_data.csv"):
        #     df = pd.DataFrame(columns=["username", "password"])
        #     df.to_csv("user_data.csv", index=False)

        # users = pd.read_csv("user_data.csv")
        # if username in users["username"].values:
        #     st.warning("Username already exists.")
        # else:
        #     new_user = pd.DataFrame([[username, hashed_pw]], columns=["username", "password"])
        #     new_user.to_csv("user_data.csv", mode='a', header=False, index=False)
        #     st.success("Account created! Please go to the Login page.")
