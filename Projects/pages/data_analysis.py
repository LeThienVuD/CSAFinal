import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import warnings
import sys
import streamlit as st
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

def data_analysis():
    # Title of the app
    st.title("Customer Segmentation and Analysis")

    st.subheader("Load and Prepare the Dataset")

    # Load dataset
    file_path = './data/marketing_campaign.csv'  # Relative path within the project

    # Read the file
    data = pd.read_csv(file_path, sep = ",")

    st.image('./images/Dataset explanation.png')  # Display image from the 'images' folder

    # Add an expander to show the raw data
    with st.expander("Show Raw Data"):
        st.write(data)

    # Remove missing values
    data = data.dropna()

    # Feature engineering (as you previously mentioned)
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format='mixed')
    dates = [i.date() for i in data["Dt_Customer"]]

    # Feature Engineering: Customer_For
    d1 = max(dates)
    days = [(d1 - i).days for i in dates]
    data["Customer_For"] = days
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

    # Feature Engineering: Age
    data["Age"] = 2025 - data["Year_Birth"]

    # Total spending on various items
    data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

    # Additional feature engineering
    data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"})
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

    # Renaming columns for clarity
    data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

    # Dropping redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)
    # Dropping outliers for Age and Income
    data = data[(data["Age"] < 90)]
    data = data[(data["Income"] < 600000)]

    # Add an expander to show the cleaned data with the newly engineered features
    with st.expander(f"Show Cleaned Data ({len(data)} records) and extra engineered features"):
        st.write(data)
        # Show the relevant columns including engineered features
        st.write(data[['Recency', 'Age', 'Spent', 'Is_Parent', 'Living_With', 'Children', 'Family_Size']])

    # Pairplot of selected features
    st.header("""Plotting some selected features""")
    To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
    fig = sns.pairplot(data[To_Plot], hue="Is_Parent", palette=["#682F2F", "#F3AB60"])
    st.subheader("Pairplot of Selected Features")
    st.pyplot(fig)
    st.write("""
    ### Pairplot Explanation:
    The pairplot visualizes the relationships between selected features in the dataset. The diagonal plots show the distribution of each feature, while the off-diagonal plots display the pairwise relationships between features.

    **Key Features**:
    - **Income**: Represents the customer's income.
    - **Recency**: The number of days since the customer's last purchase.
    - **Customer_For**: The number of days since the customer enrolled.
    - **Age**: The customer's age.
    - **Spent**: The total amount spent by the customer.
    - **Is_Parent**: Whether the customer is a parent (1) or not (0).

    **Hue (Is_Parent)**:
    - The **hue** parameter is set to `Is_Parent`, so customers who are parents are plotted in one color (e.g., `#682F2F`), and non-parents in another (e.g., `#F3AB60`).
    - This helps you visually distinguish between the two groups and analyze how the variables differ between them.
    """)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    # Exclude non-numeric columns before calculating correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])
    # Calculate correlation matrix on numeric columns only
    corrmat = numeric_data.corr()
    fig_corr = plt.figure(figsize=(20,20))
    sns.heatmap(corrmat, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig_corr)
    st.write("""
    ### Correlation Matrix Explanation:
    The correlation matrix shows the relationships between numeric features in the dataset. 
    The values range from -1 to 1:
    
    - **Positive correlation** (close to 1): Indicates that as one feature increases, the other also increases.
    - **Negative correlation** (close to -1): Indicates that as one feature increases, the other decreases.
    - **No correlation** (close to 0): Indicates no meaningful relationship between the features.

    **Key Insights**:
    - **Income and Spent**: Positive correlation may suggest that higher-income individuals tend to spend more.
    - **Age and Customer_For**: Negative correlation, as older customers may have been enrolled for a longer time.
    - **Is_Parent and Spent**: You might see a spending difference between parents and non-parents.
    """)

    st.header("""Data preprocessing""")
    # Get list of categorical variables
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)

    # Label Encoding the object dtypes.
    LE = LabelEncoder()
    for i in object_cols:
        data[i] = data[[i]].apply(LE.fit_transform)

    # Creating a copy of data
    ds = data.copy()
    # creating a subset of dataframe by dropping certain features
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    
    # Scaling
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)

    # Add an expander to show the dataframe to be used for further modeling
    with st.expander("Dataframe to be used for further modeling"):
        st.write(scaled_ds.head())

    st.header("""Dimensionality reduction""")
    # Initiating PCA to reduce dimensions
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])

    # 3D Projection Of Data In The Reduced Dimension
    x = PCA_ds["col1"]
    y = PCA_ds["col2"]
    z = PCA_ds["col3"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="maroon", marker="o")
    ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
    st.pyplot(fig)
    st.write("""
    ### 3D Projection Explanation:
    This 3D scatter plot visualizes the dataset after applying PCA for dimensionality reduction to three components (`col1`, `col2`, `col3`).
    Each point in the plot represents a customer in the reduced 3D space.
    """)

    st.header("""Clustering""")
    # Quick examination of elbow method to find the number of clusters
    st.write('Elbow Method to determine the number of clusters to be formed:')

    # Manual Elbow Method with fit time
    X = PCA_ds.values  # Get the data in matrix form
    distortions = []
    fit_times = []
    K_range = range(1, 11)

    for k in K_range:
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        end_time = time.time()

        distortions.append(kmeans.inertia_)
        fit_times.append(end_time - start_time)

    # Create the plot
    fig_elbow, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(K_range, distortions, marker='o', color='tab:blue')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Instantiate a second y-axis for fit time
    ax2.plot(K_range, fit_times, marker='o', color='tab:red')
    ax2.set_ylabel('Fit time (seconds)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add a vertical line at k=4 (or wherever the elbow is)
    ax1.axvline(x=4, color='gray', linestyle='--')

    ax1.set_title('Elbow Method for Optimal k with Fit Time')

    st.pyplot(fig_elbow)
    st.write("""
    ### Elbow Method with Fit Time Explanation:
    - The **blue line** shows the **inertia** (within-cluster sum of squared distances).
    - The **red line** shows the **fit time** (in seconds) for each value of `k`.
    - The **vertical gray line** at `k=4` indicates the chosen optimal number of clusters.
    """)

    # Initiating the Agglomerative Clustering model 
    AC = AgglomerativeClustering(n_clusters=4)
    yhat_AC = AC.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = yhat_AC
    data["Clusters"] = yhat_AC

    # Plotting the clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
    ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
    ax.set_title("The Plot Of The Clusters")
    st.pyplot(fig)
    st.write("""
    ### Cluster Plot Explanation:
    This 3D scatter plot shows the clustering results of Agglomerative Clustering with 4 clusters.
    Each point is colored based on its assigned cluster, and the plot provides insights into the distribution of clusters in the 3D reduced space.
    """)

    st.header("""Evaluating models""")

    # Cluster Distribution Plot
    st.subheader("Cluster Distribution")
    pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a 2D subplot
    pl = sns.countplot(x=data["Clusters"], palette=pal, ax=ax)  # Correct use of countplot
    pl.set_title("Distribution Of The Clusters")
    st.pyplot(fig)  # Display the plot in Streamlit
    st.write("""
    ### Cluster Distribution Explanation:
    The count plot shows how the data is distributed across the different clusters. 
    Each bar represents the number of data points (customers) in a specific cluster. 
    This allows you to see how many customers belong to each cluster after applying Agglomerative Clustering.
    """)

    # Cluster Profile Based on Income and Spending
    st.subheader("Cluster's Profile Based On Income And Spending")
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a 2D subplot
    pl_1 = sns.scatterplot(data=data, x="Spent", y="Income", hue="Clusters", palette=pal, ax=ax)
    pl_1.set_title("Cluster's Profile Based On Income And Spending")
    plt.legend()  # Add a legend to the scatter plot
    st.pyplot(fig)  # Display the plot in Streamlit
    st.write("""
    ### Cluster Profile Explanation:
    This scatter plot helps visualize how customers in different clusters are distributed based on their spending (Spent) and income (Income). 
    Each point represents a customer, and the color corresponds to the cluster they belong to.
    """)

    # Swarmplot and Boxplot for Spending (Spent) Distribution Across Clusters
    st.subheader("Spending Distribution Across Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a 2D subplot
    plt.figure(figsize=(8, 6))
    pl_2 = sns.swarmplot(x=data["Clusters"], y=data["Spent"], color="#CBEDDD", alpha=0.5, ax=ax)
    pl_3 = sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal, ax=ax)
    plt.show()  # Both plots will appear in the same figure
    st.pyplot(fig)  # Display the plot in Streamlit
    st.write("""
    ### Spending Distribution Explanation:
    The combination of a swarmplot and boxenplot shows the distribution of customer spending (Spent) across clusters. 
    - The **swarmplot** shows individual data points.
    - The **boxenplot** shows the distribution, including the median and interquartile ranges.
    This helps to compare how spending varies between clusters.
    """)

    # Countplot for Total Promotions Accepted Based on Clusters
    st.subheader("Total Promotions Accepted by Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a 2D subplot
    data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]
    pl_4 = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal, ax=ax)
    pl_4.set_title("Count Of Promotion Accepted")
    pl_4.set_xlabel("Number Of Total Accepted Promotions")
    st.pyplot(fig)  # Display the plot in Streamlit
    st.write("""
    ### Promotion Acceptance Explanation:
    This count plot shows the distribution of the number of promotions accepted by customers across different clusters. 
    Each bar represents the number of customers who accepted a specific number of promotions, and the color indicates the cluster.
    """)

    # Boxenplot for Number of Deals Purchased by Clusters
    st.subheader("Number of Deals Purchased Across Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a 2D subplot
    pl_5 = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal, ax=ax)
    pl_5.set_title("Number of Deals Purchased")
    st.pyplot(fig)  # Display the plot in Streamlit
    st.write("""
    ### Deals Purchased Explanation:
    This boxenplot shows the distribution of the number of deals purchased by customers in each cluster. 
    It provides insights into how each cluster responds to different promotional deals.
    """)

    st.header("""Profiling""")

    # Personal features for profiling
    Personal = ["Kidhome", "Teenhome", "Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education", "Living_With"]

    # Loop through each personal feature and create the jointplot for spending
    for i in Personal:
        st.subheader(f"Jointplot: {i} vs. Spending")
        # Create the jointplot for each personal feature vs Spending, grouped by clusters
        fig, ax = plt.subplots(figsize=(8, 6))
        jointplot = sns.jointplot(x=data[i], y=data["Spent"], hue=data["Clusters"], kind="kde", palette=pal)
        st.pyplot(jointplot.fig)  # Display the plot in Streamlit

        # Description of each plot
        st.write(f"""
        ### Plot Explanation for {i}:
        This jointplot shows the relationship between the feature `{i}` and the total amount spent by customers. 
        The plot uses a **Kernel Density Estimation (KDE)** method to visualize the distribution of spending for each cluster. 
        The **hue** is set to `Clusters` to visualize how customers from different clusters are distributed based on this feature and their spending behavior.
    
        - **X-axis**: Represents the values of the feature `{i}`.
        - **Y-axis**: Represents the amount of money spent by the customers (`Spent`).
        - The **KDE curves** indicate the density of spending across the range of the feature `{i}`.
    
        This visualization helps in understanding how different clusters behave with respect to this feature and how it influences spending.
        """)
    st.image('images/Dataset insights.png')  # Display image from the 'images' folder