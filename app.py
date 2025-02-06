import streamlit as st

# Set up the page configuration (optional)
st.set_page_config(page_title="Customer Segmentation and Analysis", page_icon="📊")

# Add the theme toggle button inside the sidebar (similar to being part of the navigation)
st.sidebar.title("Navigation")
theme_mode = st.sidebar.radio("Select Theme Mode", ('Dark Mode', 'Light Mode'))

# Inject CSS for Dark Mode with Neon Green accents or Light Mode
if theme_mode == "Dark Mode":
    st.markdown("""
        <style>
        /* General settings for Dark Mode */
        body {
            background-color: #1e1e1e; /* Dark background */
            color: #f5f5f5; /* Light text color for body */
        }

        /* Custom styles for stMain (main content area) */
        [data-testid="stMain"] {
            background-color: #1e1e1e; /* Dark background */
            color: #f5f5f5; /* Light text color for body */
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #2e2e2e; /* Sidebar dark background */
            /* color: #f5f5f5;  Sidebar text color */
            color: #39ff14
        }

        /* Sidebar Titles and Labels */
        [data-testid="stSidebar"] #stSidebarHeader,
        [data-testid="stSidebar"] #stSidebarTitle,
        [data-testid="stSidebar"] #stSidebarMenu {
            color: #f5f5f5; /* Light text for sidebar */
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] button {
            color: #f5f5f5; /* Light text for sidebar buttons */
        }

        /* Text in the main content */
        h1, h2, h3, h4, h5, h6 {
            color: #f5f5f5; /* Light text for paragraphs and headings */
        }

        /* Header */
        [data-testid="stHeader"] {
            background-color: #F4D03F;
            background-image: linear-gradient(132deg, #F4D03F 0%, #16A085 100%); /* Dark header background */
            color: #39ff14; /* Neon Green header text */
        }

        /* Button styles */
        div[data-testid="stBaseButton"] > button {
            background-color: #39ff14; /* Neon Green */
            color: black; /* Button text color */
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
        }

        div[data-testid="stBaseButton"] > button:hover {
            background-color: #32e100; /* Lighter neon green for hover */
        }

        /* Target "Select Theme Mode" and "Go to" texts */
        /* Apply neon green color to text in markdown containers */
        [data-testid="stMarkdownContainer"] > h1, h2, h3 {
            color: #39ff14 !important; /* Neon Green */
        }

        </style>
    """, unsafe_allow_html=True)

else:  # Light Mode
    st.markdown("""
        <style>
        /* General settings for Light Mode */
        body {
            background-color: #ffffff; /* Light background */
            color: #333333; /* Dark text color for body */
        }

        /* Custom styles for stMain (main content area) */
        [data-testid="stMain"] {
            background-color: #ffffff; /* Light background */
            color: #333333; /* Dark text color for body */
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f4f4f4; /* Sidebar light background */
            color: #333333; /* Sidebar text color */
        }

        /* Sidebar Titles and Labels */
        [data-testid="stWidgetLabel"] > div > p {
            color: #333333; /* Dark text for sidebar text */
        }
        
        [data-testid="stRadio"] > div > label > div > div > p {
            color: #333333; /* Light Blue text for sidebar */
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] button {
            color: #333333; /* Dark text for sidebar buttons */
        }

        /* Text in the main content */
        h1, h2, h3, h4, h5, h6 {
            color: #333333; /* Dark text for paragraphs and headings */
        }

        /* Header */
        [data-testid="stHeader"] {
            background-color: #8BC6EC;
            background-image: linear-gradient(135deg, #8BC6EC 0%, #9599E2 100%); /* Light header background */
            color: #4f8ef7; /* Light Blue header text */
        }

        /* Button styles */
        div[data-testid="stBaseButton"] > button {
            background-color: #4f8ef7; /* Light Blue */
            color: black; /* Button text color */
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
        }

        div[data-testid="stBaseButton"] > button:hover {
            background-color: #4682ff; /* Lighter blue for hover */
        }

        /* Target "Select Theme Mode" and "Go to" texts */
        /* Apply blue color to text in markdown containers */
        [data-testid="stMarkdownContainer"] > h1, h2, h3 {
            color: #4f8ef7 !important; /* Light Blue */
        }

        </style>
    """, unsafe_allow_html=True)

# Remove default navigation bar and add a new style for sidebar hiding
no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Go to", ["Homepage", "Data Analysis", "Insights & Conclusion"])

# Conditionally import and render the pages based on the sidebar selection
if page == "Homepage":
    from pages.homepage import homepage
    homepage()  # Call the Homepage function
elif page == "Data Analysis":
    from pages.data_analysis import data_analysis
    data_analysis()  # Call the Data_analysis function
elif page == "Insights & Conclusion":
    from pages.conclusion import insights_and_conclusion
    insights_and_conclusion()  # Call the Data_analysis function
