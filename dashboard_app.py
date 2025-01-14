import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    file_path = "supermarket_sales.csv"
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.time
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Weekday'] = data['Date'].dt.weekday
        data['Hour'] = data['Time'].apply(lambda x: x.hour)
        # Map ratings to sentiments
        data['Sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x >= 7 else 'Neutral' if 4 <= x < 7 else 'Negative')
        return data
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}. Please ensure it is in the correct directory.")
        return None

data = load_data()

# App Title
st.title("Business-Focused Dashboard for Grant Application")

# Business Usefulness Context
st.markdown(
    """
    ### Why This Dashboard Matters to Businesses
    This interactive dashboard provides critical insights to help businesses make informed decisions. Features include:
    - **Sales Forecasting**: Predict future revenue and identify trends to optimize business strategy.
    - **Customer Segmentation**: Understand customer behavior to tailor marketing efforts and improve customer retention.
    - **Inventory Prediction**: Forecast inventory needs to reduce waste and avoid stockouts.
    - **Exploratory Data Analysis**: Identify key business performance metrics.
    
    By leveraging these insights, businesses can enhance efficiency, profitability, and customer satisfaction.
    """
)

# Dropdown Menu
menu = st.selectbox(
    "Choose a dashboard section:",
    [
        "Select an option", 
        "Sales Forecasting", 
        "Customer Segmentation", 
        "Inventory Prediction", 
        "Exploratory Data Analysis"
    ]
)

# Display Results Based on Menu Selection
if data is not None:
    if menu == "Sales Forecasting":
        st.header("Sales Forecasting")
        # Group and plot sales over time
        daily_sales = data.groupby('Date')['Total'].sum()
        st.line_chart(daily_sales)

        # Decompose the time series to understand trends
        st.subheader("Time Series Decomposition")
        daily_sales_ts = daily_sales.reset_index().set_index('Date')['Total']
        decomposition = seasonal_decompose(daily_sales_ts, model='additive', period=7)
        fig = decomposition.plot()
        st.pyplot(fig)

        st.markdown(
            """
            #### Business Impact
            Forecasting sales trends allows businesses to:
            - Plan future revenue strategies.
            - Identify peak seasons to adjust marketing and staffing efforts.
            - Improve cash flow management.
            """
        )

    elif menu == "Customer Segmentation":
        st.header("Customer Segmentation")
        # Prepare data for clustering
        segmentation_data = data[['Total', 'Quantity', 'Rating']]
        scaler = StandardScaler()
        segmentation_data_scaled = scaler.fit_transform(segmentation_data)

        # Use KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        segmentation_data['Cluster'] = kmeans.fit_predict(segmentation_data_scaled)
        data['Customer_Cluster'] = segmentation_data['Cluster']

        # Add cluster descriptions
        cluster_descriptions = segmentation_data.groupby('Cluster').mean()
        st.subheader("Cluster Descriptions")
        st.write(cluster_descriptions)

        # Visualize clusters
        st.subheader("Customer Clusters")
        pca = PCA(n_components=2)
        segmentation_data_pca = pca.fit_transform(segmentation_data_scaled)
        fig, ax = plt.subplots()
        for cluster in range(3):
            cluster_points = segmentation_data_pca[segmentation_data['Cluster'] == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
        ax.set_title('Customer Segmentation Clusters', fontsize=16)
        ax.set_xlabel('PCA Component 1', fontsize=12)
        ax.set_ylabel('PCA Component 2', fontsize=12)
        ax.legend()
        st.pyplot(fig)

        st.markdown(
            """
            #### Business Impact
            Customer segmentation enables businesses to:
            - Identify high-value customers and focus marketing efforts.
            - Personalize promotions to specific customer groups.
            - Enhance customer retention by addressing unique needs.
            """
        )

    elif menu == "Inventory Prediction":
        st.header("Inventory Prediction")
        # Prepare data for regression
        inventory_data = pd.get_dummies(data, columns=['Product line', 'Sentiment'], drop_first=True)
        X = inventory_data.drop(['Invoice ID', 'Date', 'Time', 'Branch', 'City', 'Customer type',
                                 'Gender', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating', 'Quantity'], axis=1)
        y = inventory_data['Quantity']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Gradient Boosting Regressor
        gb_model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=3)
        gb_model.fit(X_train, y_train)
        y_pred = gb_model.predict(X_test)

        # Display evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Absolute Error: {mae:.3f}")
        st.write(f"R² Score: {r2:.3f}")

        # Feature Importance Visualization
        st.subheader("Feature Importances")
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': gb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importances.set_index('Feature'))

        # Future Demand Prediction Example
        st.subheader("Predict Future Inventory Needs")
        example_input = {feature: 0 for feature in X.columns}
        example_input['Total'] = st.number_input("Enter total sales amount", min_value=0.0, value=500.0, step=50.0)
        example_input['Tax 5%'] = st.number_input("Enter tax amount (5%)", min_value=0.0, value=25.0, step=5.0)
        example_input_df = pd.DataFrame([example_input])
        predicted_quantity = gb_model.predict(example_input_df)[0]
        st.write(f"Predicted Inventory Need: {predicted_quantity:.2f} units")

        # Visualization for future inventory needs
        st.subheader("Future Inventory Needs Visualization")
        future_predictions = []
        for i in range(5):
            example_input['Total'] += 100  # Simulate an increase in sales
            example_input_df = pd.DataFrame([example_input])
            future_predictions.append(gb_model.predict(example_input_df)[0])

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 6), future_predictions, marker='o', label='Predicted Inventory')
        plt.title('Predicted Inventory Needs Over Time', fontsize=16)
        plt.xlabel('Future Periods', fontsize=12)
        plt.ylabel('Inventory Needs', fontsize=12)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        st.markdown(
            """
            #### Business Impact
            Predicting inventory needs helps businesses to:
            - Reduce waste by avoiding overstocking.
            - Prevent stockouts to maintain customer satisfaction.
            - Optimize inventory management costs.
            """
        )

    elif menu == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        # Sales by Product Line
        product_line_sales = data.groupby('Product line')['Total'].sum()
        st.bar_chart(product_line_sales)

        # Sentiment Distribution
        sentiment_distribution = data['Sentiment'].value_counts()
        st.bar_chart(sentiment_distribution)

        st.markdown(
            """
            #### Business Impact
            EDA highlights key metrics such as:
            - Top-performing product lines to focus efforts.
            - Customer sentiment analysis to understand satisfaction.
            """
        )
