import pickle
from collections import Counter
from io import BytesIO
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, cm
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans

# Set the backend of matplotlib
matplotlib.use('Agg')

if 'random_values' not in st.session_state:
    st.session_state.random_values = pd.DataFrame()

def load_data():
    """Load the main dataset and insights."""
    df = pd.read_csv('data/CC GENERAL.csv')
    insights_df = pd.read_csv('data/insights.csv')
    return df, insights_df

def process_data(df):
    missing_var = [var for var in df.columns if df[var].isnull().sum() > 0]
    # Preprocess the data - Impute missing values if necessary
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())

    df = df.drop("CUST_ID", axis=1)
    return (df)

def score_elbow(df, pca_df):
    '''
    pca = PCA(n_components=8)
    principal_comp = pca.fit_transform(df)

    pca_df = pd.DataFrame(data=principal_comp, columns=["pca1", "pca2", "pca3", "pca4", "pca5", "pca6", "pca7", "pca8"])
    '''
    kmeans = KMeans(random_state=0)

    # Instantiate the KElbowVisualizer with the KMeans model and the range of clusters to explore
    visualizer = KElbowVisualizer(kmeans, k=(1, 10))
    # Fit the data to the visualizer
    visualizer.fit(pca_df)
    # Finalize and render the figure internally, then show it using Streamlit
    visualizer.finalize()
    st.pyplot(visualizer.fig)
    return kmeans

@st.cache_data
def scatter_plot(pca_df, cluster_labels, title):
    """
    Create a scatter plot for clustering results.
    Parameters:
    - pca_df: DataFrame containing PCA components
    - cluster_labels: Cluster labels for each point
    - title: Title for the plot
    """
    # Concatenate PCA DataFrame with cluster labels
    pca_df_clustered = pd.concat([pca_df, pd.DataFrame({'cluster': cluster_labels})], axis=1)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define a dynamic palette
    n_clusters = len(np.unique(cluster_labels))
    palette = sns.color_palette("hsv", n_clusters)

    # Create the scatter plot
    sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df_clustered, palette=palette, ax=ax)

    # Set plot title
    ax.set_title(title)

    # Display the plot in Streamlit
    st.pyplot(fig)

@st.cache_data
def silhouette_plot(X, cluster_labels, title):
    """
    Create a silhouette plot for any clustering method.
    Parameters:
    - X: The dataset used for clustering
    - cluster_labels: Cluster labels for each point
    - title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    silhouette_vals = silhouette_samples(X, cluster_labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(cluster_labels)):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        color = cm.jet(float(i) / np.max(cluster_labels + 1))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    ax.set_title(title)
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster labels')
    ax.set_yticks(y_ticks, labels=np.unique(cluster_labels))
    ax.axvline(x=silhouette_vals.mean(), color="red", linestyle="--")

    st.pyplot(fig)

def elbow_plot(df, pca_df, gmm):
    pass

def cluster_analysis_options(df):
    """Options for cluster analysis in the sidebar."""
    st.sidebar.subheader("Cluster Analysis Options")
    cluster_method = st.sidebar.selectbox('Select Cluster Method',
                                          ('Kmeans', 'Gaussian Mixture', 'Agglomerative Nesting'),
                                          key='cluster_method')
    # PCA transformation
    pca = PCA(n_components=8)
    principal_comp = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=principal_comp, columns=["pca1", "pca2", "pca3", "pca4", "pca5", "pca6", "pca7", "pca8"])

    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4, key='n_clusters')

    st.subheader("Cluster Analysis")
    st.write(f"Selected Method: {cluster_method}")
    st.write(f"Number of Clusters: {n_clusters}")

    # Initialize cluster labels
    cluster_labels = None

    # Clustering analysis based on the selected method
    if cluster_method == 'Kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(df)

        with st.expander("Distortion Score Elbow for KMeans Clustering"):
            score_elbow(df, pca_df)

    elif cluster_method == 'Gaussian Mixture':
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        cluster_labels = gmm.fit_predict(df)

    elif cluster_method == 'Agglomerative Nesting':
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = agglomerative.fit_predict(df)

    # Scatter Plot for all methods
    with st.expander("Scatter Plot"):
        title = f"Clustering using {cluster_method} Algorithm"
        scatter_plot(pca_df, cluster_labels, title)

    # Silhouette Plot for all methods
    with st.expander("Silhouette Plot"):
        title = f"Silhouette Plot for {cluster_method} Clustering"
        silhouette_plot(df, cluster_labels, title)  # Assuming df is the correct input here; adjust as necessary

def exploratory_data_analysis(df, insights_df, df_scaled):
    """Display the Exploratory Data Analysis Section."""
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown(
        "Explore the dataset to understand the distribution of various features and their relation to customer churn.")

    # Various expanders for EDA
    with st.expander("Preview Dataset"):
        number = st.number_input("Number of Rows to Show", min_value=5, max_value=100, value=10)
        st.dataframe(df.head(number))

    with st.expander("Show Descriptive Statistics"):
        st.write(df.describe())

    plot_histograms(df, insights_df)

    with st.expander("Distribution of Tenure"):
        plot_tenure_distribution(df)

    # Use an expander for outliers value counts of a selected column
    plot_outliers(df)

    # Use an expander for the correlation matrix heatmap
    with st.expander("Show Correlation Matrix Heatmap"):
        plot_Heatmap(df_scaled)

def plot_histograms(df, insights_df):
    """Plot histograms for selected columns and show insights."""
    with st.expander("Plot histograms for each selected column"):
        columns_of_interest = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                               'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                               'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                               'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                               'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

        insights_df = insights_df[columns_of_interest].dropna(how='all')
        selected_columns = st.multiselect('Select columns to plot', columns_of_interest)

        if selected_columns:
            for column in selected_columns:
                st.write(f"Histogram for {column}")
                fig, ax = plt.subplots()
                ax.hist(df[column].dropna(), bins=20, alpha=0.7)
                ax.set_title(f'Histogram of {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

                if column in insights_df.columns:
                    insight = insights_df[column].iloc[0]
                    st.write(f"Insight for {column}: {insight}")
                else:
                    st.write(f"No insights available for {column}.")
        else:
            st.write("No columns selected")


def plot_tenure_distribution(df):
    # Plotting / visualize and analyze - Balance Vs Tenure

    plt.figure(figsize=(6, 4))
    ax = plt.axes()
    # ax.set_facecolor('darkgrey')
    sns.violinplot(x='TENURE', y='BALANCE', data=df, inner='quartile')
    plt.xlabel('ACCOUNT TENURE')
    plt.ylabel('ACCOUNT BALANCE')
    plt.title('ACCOUNT BALANCE OVER TENURE')
    st.pyplot(plt)

    # Plot countplot
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='TENURE')

    # Calculate percentages
    total = len(df['TENURE'])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height / total:.1%}',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8)

    plt.title('Distribution of TENURE')
    st.pyplot(plt)  # Use st.pyplot to display the figure in Streamlit
    st.write("Most customers (84\%) the tenure of credit card service are 12 months.")

@st.cache_data
def plot_Heatmap(df):
    # Correlation Matrix Heatmap Visualisation
    corr_matrix = df.corr()
    # Explicitly create a figure and axes object
    fig, ax = plt.subplots(figsize=(16, 10))
    # Use seaborn to create the heatmap on the created axes
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    # Pass the figure to streamlit for rendering
    st.pyplot(fig)

def remove_outliers(df, columns):
    # Outlier Removal using Modified IQR Method:
    # This method applies an aggressive approach to remove outliers by calculating the Interquartile Range (IQR)
    # based on the 5th and 95th percentiles (Q1 and Q3, respectively). It identifies outliers as those values
    # that fall below (Q1 - 1.5 * IQR) or above (Q3 + 1.5 * IQR), focusing on eliminating extreme variations in the data.
    # This broader percentile range for Q1 and Q3 helps in retaining a significant portion of the dataset
    # while effectively filtering out the most extreme outliers, making it suitable for datasets with
    # a wide distribution or long tails.

    for column in columns:
        Q1 = df[column].quantile(0.05)
        Q3 = df[column].quantile(0.95)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df

def plot_outliers(df):
    """Plot outliers for selected columns and show insights."""
    with st.expander("Plot Outliers for each selected column"):
        columns_of_interest = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                               'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                               'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                               'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                               'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
        columns_to_filter = [
            'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
            'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS'
        ]

        # Apply the function to your DataFrame
        df_filtered = remove_outliers(df, columns_to_filter)

        # Checkbox to toggle outlier handling
        handle_outliers = st.checkbox('Handle Outliers', key='handle_outliers_checkbox')

        # Use a unique key for this multiselect widget
        selected_columns = st.multiselect('Select columns to plot', columns_of_interest,
                                          key='plot_outliers_multiselect')
        if selected_columns:
            for column in selected_columns:
                fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize to change the plot size

                if handle_outliers:
                    df = df_filtered.copy()

                sns.boxplot(x=df[column], linewidth=1.0, palette='Blues')
                ax.set_title(f'Outliers in {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Value')
                st.pyplot(fig)  # Use st.pyplot to display the figure in Streamlit
        else:
            st.write("No columns selected")


def scale_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # Convert the NumPy array back into a pandas DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

def generate_random_samples(df, n_samples=10):
    """
    Generate random samples based on the dataframe's numerical features.

    Args:
    - df: Pandas DataFrame from which to derive statistical properties.
    - n_samples: Number of random samples to generate.

    Returns:
    - A new DataFrame with n_samples random entries.
    """
    # Dropping non-numerical columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Handling missing values in the original dataset
    # For simplicity, fill missing values with the mean of the column
    df_numeric = df_numeric.fillna(df_numeric.mean())

    # Generating random samples based on mean and std of the columns
    random_samples = {}
    for column in df_numeric.columns:
        mean = df_numeric[column].mean()
        std = df_numeric[column].std()

        # Generating random data for the column
        random_samples[column] = np.random.normal(loc=mean, scale=std, size=n_samples)

    return pd.DataFrame(random_samples)


def plot_clusters(new_labels, cluster_method):
    cluster_counts = Counter(new_labels)

    # Sorting clusters to ensure the plot is ordered
    sorted_clusters = sorted(cluster_counts.items())

    # Unzipping the sorted items into two lists
    clusters, counts = zip(*sorted_clusters)

    if cluster_method == 'Kmeans':
        # Cluster descriptions as per your mapping
        cluster_descriptions = [
            "Moderate Use, Newer Customers",
            "High Balance, Credit-Focused Users",
            "Cash Advance Users with Longer Tenure",
            "High-Spending Active Users"
        ]
        # Replacing cluster numbers with descriptions for Kmeans
        labels = [cluster_descriptions[cluster] for cluster in clusters]
    else:
        # For other methods, use the original cluster labels
        labels = [f'Cluster {cluster}' for cluster in clusters]

    # Creating the plot
    plt.figure(figsize=(10, 10))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customer')
    plt.title('Number of Customer in Each Cluster')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    #st.write("Plot")
    # Show the plot
    st.pyplot(plt)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def predict_clusters(data_scaled, cluster_method):
    if cluster_method == 'Kmeans':
        model = load_model('models/kmeans_model.pkl')
    elif cluster_method == 'Gaussian Mixture':
        model = load_model('models/gmm_model.pkl')
    elif cluster_method == 'Agglomerative Nesting':
        model = load_model('models/knn_model_for_agglo.pkl')
    else:
        raise ValueError("Invalid clustering algorithm selected.")

    new_labels = model.predict(data_scaled)
    return new_labels

def render_cluster_info(new_labels, cluster_descriptions, cluster_details,cluster_method):
    cluster_counts = Counter(new_labels)
    for cluster, count in cluster_counts.items():
        description = cluster_descriptions.get(cluster, "Cluster not found")
        details = cluster_details.get(cluster, "Details not available")
        if cluster_method == 'Kmeans':
            description = cluster_descriptions.get(cluster, "Cluster not found")
            details = cluster_details.get(cluster, "Details not available")
            st.markdown(f"""
                                    <div style="background-color:#fce7e3; padding:10px; border-radius:10px; margin:3px">
                                        <h6 style="color:#333; text-align:left;">{description}: {count} customers</h6>                               
                                    </div>
                                    <div>{details}</div>
                                    """, unsafe_allow_html=True)
        else:
            # For any other cluster method, print the original label and the number of customers
            st.markdown(f"""
                                    <div style="background-color:#fce7e3; padding:10px; border-radius:10px; margin:3px">
                                        <h6 style="color:#333; text-align:left;">Cluster {cluster}: {count} customers</h6>
                                    </div>
                                    """, unsafe_allow_html=True)

def cluster_assignment_analysis(df, df_scaled):
    st.sidebar.subheader("Prediction Model Selection")
    cluster_method = st.sidebar.selectbox('Choose Clustering Algorithm',
                                          ('Kmeans', 'Gaussian Mixture', 'Agglomerative Nesting'), key='model_method')

    if cluster_method == 'Kmeans':
        cluster_descriptions, cluster_details = get_cluster_info()
    else:
        cluster_descriptions = {}
        cluster_details = {}

    if st.button('Generate Random CSV File'):
        random_df = generate_random_samples(df, n_samples=100)  # Generates 100 samples
        st.session_state['random_df'] = random_df  # Store in session state
        csv = random_df.to_csv(index=False)
        b_csv = BytesIO(csv.encode())  # Convert to bytes
        st.download_button(label="Download Random Data as CSV", data=b_csv, file_name="random_data.csv",
                           mime="text/csv")

    # Option to use generated data directly
    if 'random_df' in st.session_state and st.checkbox('Use Generated Data for Prediction'):
        new_labels = upload_and_predict(cluster_method, data=st.session_state['random_df'])
        if new_labels is not None:
            render_cluster_info(new_labels, cluster_descriptions, cluster_details, cluster_method)
            plot_clusters(new_labels, cluster_method)

    else:
        new_labels = upload_and_predict(cluster_method)
        if new_labels is not None:
            render_cluster_info(new_labels, cluster_descriptions, cluster_details, cluster_method)
            plot_clusters(new_labels, cluster_method)

# ----------------------------------------------------
def upload_and_predict(cluster_method, data=None):
    if data is not None:
        data_scaled = scale_data(data)
        new_labels = predict_clusters(data_scaled, cluster_method)
        return new_labels
    else:
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data_scaled = scale_data(data)
            new_labels = predict_clusters(data_scaled, cluster_method)
            return new_labels
    return None


def get_cluster_info():
    # Cluster descriptions, tooltips, and details
    cluster_descriptions = {
        0: "Cluster 0: Moderate Use, Balanced Borrowers",
        1: "Cluster 1: Active Spenders & Loyalists",
        2: "Cluster 2: Cautious Participants",
        3: "Cluster 3: High-Spending Active Users - Installment Savers"
    }

    # Extended descriptions for each cluster
    cluster_details = {
        0: """
               <ul>
                   <li>Purchase Behavior: Moderate engagement in installment purchases with a moderate frequency of one-off purchases.</li>
                   <li>Credit Limit: Higher credit limits compared to other clusters.</li>
                   <li>Cash Advances: Moderate to high usage frequency, indicating reliance on cash advances.</li>
                   <li>Payment Habits: Less frequent transactions but with higher payment amounts.</li>
                   <li>Tenure: Long-standing customer relationships with a mean tenure around 11 years.</li>
               </ul>
               """,
        1: """
               <ul>
                   <li>High frequency of both one-off and installment purchases, indicating active spending.</li>
                   <li>High credit limits, similar to Cluster 0.</li>
                   <li>Cash Advances: Infrequent usage, suggesting a lower reliance on cash advances.</li>
                   <li>Payment Habits: Active in making payments, potentially indicating good financial management.</li>
                   <li>Tenure: Very loyal customers with the longest average tenure, slightly above 11 years.</li>
               </ul>
               """,
        2: """
               <ul>
                   <li>Purchase Behavior: Lowest engagement in both one-off and installment purchases.</li>
                   <li>Credit Limit: Lowest credit limits among all clusters.</li>
                   <li>Cash Advances: Low frequency of usage, indicating cash advances are rarely used.</li>
                   <li>Payment Habits: Minimal activity, possibly due to lower credit limits.</li>
                   <li>Tenure: Long tenure, around 11 years, despite lower financial activity.</li>
               </ul>
               """,
        3: """
               <ul>
                   <li>Purchase Behavior: Prefer installment purchases significantly over one-off purchases.</li>
                   <li>Credit Limit: Moderate credit limits, not as high as Clusters 0 and 1.</li>
                   <li>Cash Advances: Very low frequency of usage, the least reliant on cash advances.</li>
                   <li>Payment Habits: Payment frequency is not the highest but shows consistent behavior.</li>
                   <li>Tenure: Loyalty is apparent with long tenure, averaging around 11.5 years.</li>
               </ul>
               """
    }
    return cluster_descriptions, cluster_details

def main():
    """Main function to construct the Streamlit app."""
    st.title('Customer Segmentation for Marketing Strategy')
    st.sidebar.title("Navigation")
    activity = ["Exploratory Data Analysis", "Model Analysis", "Cluster Assignment Prediction"]
    choice = st.sidebar.radio("Choose an Activity", activity)

    df, insights_df = load_data()
    data = df.copy()
    df = process_data(df)
    df_scaled = scale_data(df)
    # Tooltip based on selection
    tooltips = {
        "Exploratory Data Analysis": "Dive into the dataset to visualize and understand the distribution, trends, and patterns of various features, laying the groundwork for deeper analysis.",
        "Model Analysis": "Group data points into clusters based on similarity, using algorithms like K-means, Hierarchical, Gaussian Mixture, or DBSCAN to identify patterns and relationships.",
        "Cluster Assignment Prediction": "Predict the most likely cluster assignment for new data points based on the clustering model, enabling targeted strategies or personalized interventions."
    }

    # Display the tooltip for the current selection
    st.sidebar.markdown(f"*{tooltips[choice]}*", unsafe_allow_html=True)

    # Conditional content based on sidebar choice
    if choice == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        exploratory_data_analysis(df, insights_df, df_scaled)
    elif choice == 'Model Analysis':
        # st.subheader("Cluster Analysis")
        cluster_analysis_options(df_scaled)
    elif choice == 'Cluster Assignment Prediction':
        st.subheader("Cluster Assignment Prediction")
        cluster_assignment_analysis(df, df_scaled)

if __name__ == '__main__':
    main()
