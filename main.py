import streamlit as st
from login import login
from signup import signup
import streamlit_app  # this must have a `main()` function

# Setup page state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# Sidebar menu
if st.session_state.logged_in:
    st.session_state.page = "Dashboard"  # force dashboard if already logged in

if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])
    if menu == "Login":
        st.session_state.page = "Login"
    elif menu == "Sign Up":
        st.session_state.page = "Sign Up"
else:
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    logout = st.sidebar.button("Logout")
    if logout:
        st.session_state.logged_in = False
        st.session_state.page = "Login"

# Page rendering
if st.session_state.page == "Login":
    login()
elif st.session_state.page == "Sign Up":
    signup()
elif st.session_state.page == "Dashboard" and st.session_state.logged_in:
    streamlit_app.main()
else:
    st.warning("Unauthorized. Please login.")
