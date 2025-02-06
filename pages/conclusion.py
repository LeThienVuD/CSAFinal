import streamlit as st

def insights_and_conclusion():
    # Insights Section
    st.subheader("Key Insights From The Data Analysis")

    st.write("""
    ### 1. Pairplot of Selected Features:
    The pairplot visualized the relationships between several key features like **Income**, **Recency**, **Customer_For**, **Age**, **Spent**, and **Is_Parent**. 
    - **Income vs. Spent**: A positive correlation can be seen, indicating that customers with higher income tend to spend more.
    - **Age vs. Customer_For**: There is a negative correlation between **Age** and **Customer_For**, suggesting that younger customers may have more recent enrollments.
    - **Is_Parent**: Customers who are parents tend to have distinct patterns compared to non-parents, especially in terms of their spending behavior.
    
    ### 2. Correlation Matrix:
    The heatmap of the correlation matrix showed relationships between numeric variables.
    - **Income and Spent**: There is a strong positive correlation, confirming the assumption that higher-income customers tend to spend more.
    - **Age and Customer_For**: Negative correlation implies that older customers tend to have been customers for a longer period.
    
    ### 3. Dimensionality Reduction (PCA):
    The PCA transformation reduced the dataset to three dimensions, which we visualized in a 3D scatter plot.
    - This helped us identify patterns in customer behavior based on their features in the reduced 3D space.
    
    ### 4. Elbow Method for Clustering:
    We used the **Elbow Method** to determine the optimal number of clusters for customer segmentation. 
    - The optimal number of clusters was found to be **4**, as indicated by the "elbow" point in the graph, where the inertia begins to plateau.
    
    ### 5. Clustering Results:
    The **Agglomerative Clustering** algorithm formed 4 clusters based on the reduced features.
    - The 3D scatter plot of clusters clearly shows distinct groups of customers. Each group represents customers with similar behaviors, making it easier for businesses to target different customer segments.

    ### 6. Cluster Profiles Based on Income and Spending:
    The scatter plot showed how the clusters differ in terms of **Income** and **Spending**.
    - Clusters with higher income tend to have higher spending, while other clusters show varying levels of spending based on income.

    ### 7. Spending Distribution Across Clusters:
    The **Swarmplot and Boxenplot** revealed that:
    - Some clusters have highly varied spending behaviors, while others show more consistent spending patterns.
    - The **boxenplot** highlighted that certain clusters have significantly higher spending ranges.

    ### 8. Promotion Acceptance by Clusters:
    The count plot revealed the number of promotions accepted by customers in each cluster.
    - Customers in certain clusters are more likely to accept promotions, which could be important for targeted marketing.

    ### 9. Number of Deals Purchased:
    The **boxenplot** for the number of deals purchased showed that:
    - Some clusters purchase more deals, suggesting that these customers are more engaged with promotional offers.

    ### 10. Personal Feature Profiling:
    The jointplots showed how personal features like **Age**, **Children**, and **Family Size** relate to spending.
    - Customers with children and larger families tend to spend more, which could be used to tailor product offerings.
    """)

    # Conclusion Section
    st.subheader("Conclusion and Next Steps")

    st.write("""
    ### Conclusion:
    Based on the analysis, we gained valuable insights into customer behaviors, spending patterns, and their responsiveness to promotions:
    
    - **Income and Spending**: There is a clear correlation between income and spending, suggesting that targeting high-income customers could lead to increased revenue.
    - **Customer Segmentation**: Using clustering, we identified 4 distinct customer segments, each with unique behaviors. This segmentation can help tailor marketing strategies, products, and promotions.
    - **Promotions**: Certain customer groups are more likely to accept promotions, which could be leveraged for targeted campaigns.
    - **Family-Oriented Spending**: Customers with children and larger families tend to spend more, which is an important factor for businesses when creating marketing campaigns or product offerings.

    ### Next Steps:
    - **Marketing Strategy**: Focus on high-income and family-oriented segments for targeted marketing campaigns.
    - **Promotions**: Consider increasing promotional offers for clusters that are more likely to engage with them.
    - **Customer Retention**: Implement loyalty programs for clusters with lower spending to increase their engagement and retention.

    This analysis can be further refined by integrating more features or external data sources for deeper customer profiling and behavior prediction.
    """)