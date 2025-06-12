#Importing the Libraries
import streamlit as st

def homepage():
    # Title of the app
    st.title("Customer Segmentation and Analysis")

    # Introduction text
    st.markdown("""
    ## Welcome to the Customer Segmentation and Analysis Dashboard
    
    This dashboard presents a detailed analysis of customer data to understand spending patterns, customer demographics, and other important attributes. 

    The dataset contains information about customer spending across various categories, including:
    
    - **Demographic Information**: Age, Marital Status, Education
    - **Customer Behavior**: Spending patterns in different product categories
    - **Customer's Duration with the Company**: How long customers have been enrolled
    - **Other Key Features**: Number of children, family size, etc.

    Through this analysis, we will explore:
    
    - Distribution of key customer attributes
    - Correlations between different features
    - Spending behavior and customer segmentation based on their demographics

    **Key Insights from the Analysis:**
    - **Customer Segmentation**: Segmentation based on parental status, spending, and age.
    - **Correlation Matrix**: Visualizing relationships between key numerical features.
    - **Pairwise Plots**: A detailed exploration of how different features are related, including spending and customer duration.

    This analysis helps businesses understand their customers better, allowing for more effective marketing strategies and product offerings.

    ### How to Use:
    - Navigate through the dashboard to explore the analysis results.
    - Go to the "Data Analysis" page to explore the detailed visualizations and correlations.

    Enjoy exploring the data!
    """)

    # Navigation instructions
    st.markdown("""
    #### How to Start:
    - Explore the analysis by clicking on the **"Data Analysis"** tab from the sidebar.
    - Dive deeper into the customer data to discover patterns and insights.
    """)

    st.markdown("___")
    st.markdown("**Vũ Lê Đức Thiện - CSA08**")
    st.markdown("**Version: 1.5**")