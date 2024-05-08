# Unsupervised-Machine-Learning-for-Customer-Market-Segmentation

For your second project on "Unsupervised Machine Learning for Customer Market Segmentation," let’s create a structured README file. You've already got a great foundation from the PDF report, including detailed methodology and results. Here's how you can structure the README:

---

## Unsupervised Machine Learning for Customer Market Segmentation

### Project Overview
This project explores customer market segmentation using unsupervised machine learning techniques. By applying clustering algorithms, we identify distinct customer groups, enabling targeted marketing strategies that cater to the unique preferences and behaviors of each segment.

### Live Demo
Check out the web application that demonstrates the segmentation in action: [Customer Market Segmentation Web App](https://custseg1.streamlit.app/)

### Objectives
- To segment the customer market using unsupervised learning methods.
- To analyze and visualize customer data to understand underlying patterns.
- To determine the optimal number of clusters for effective segmentation.

### Technologies Used
- Python
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn
- Tools: Jupyter Notebook, Google Colab

### Installation and Usage
1. Clone this repository.
2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the Jupyter Notebook to view the segmentation analysis and clustering process.

### Data Description
The dataset used for this project contains multiple attributes related to customer behavior and demographics, sourced from Kaggle. Key attributes include:
- **Balance**: Amount owed by the customer to the credit card company.
- **Purchases**: Total purchase amount made from the account.
- **One-Off Purchases**: Largest single purchase amount.
- **Credit Limit**: Maximum credit limit available to the customer.
- **Tenure**: Duration of the credit card service for the user.

### Methodology
The project follows the CRISP-DM methodology, encompassing business understanding, data understanding, data preparation, modeling, evaluation, and deployment. Clustering algorithms used include K-Means, Gaussian Mixture Models (GMM), and Hierarchical Clustering.

### Results
- **K-Means Clustering**: Identified as the most effective method for this dataset, providing clear segmentation with satisfactory internal consistency.
- **Model Evaluation**: Employed Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index to measure the effectiveness of the clustering.

### Contributing
Feel free to fork this repository and propose changes by submitting a pull request. We're open to any contributions or suggestions to improve the analysis and outcomes.
