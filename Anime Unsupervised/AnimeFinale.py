import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import pickle
from PIL import Image

# Function to download CSV from GitHub
@st.cache_data
def load_csv_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    csv_data = StringIO(response.text)
    return pd.read_csv(csv_data)

# URLs to CSV files
ANIME_CSV_URL = 'https://raw.githubusercontent.com/KamogeloThalakgale/Unsupervised-Learning_project/main/data/anime.csv'
TRAIN_CSV_URL = 'https://raw.githubusercontent.com/KamogeloThalakgale/Unsupervised-Learning_project/main/data/train.csv'
TEST_CSV_URL = 'https://raw.githubusercontent.com/KamogeloThalakgale/Unsupervised-Learning_project/main/data/test.csv'

# Load data
@st.cache_data
def load_data():
    anime_df = load_csv_from_github(ANIME_CSV_URL)
    train_df = load_csv_from_github(TRAIN_CSV_URL)
    test_df = load_csv_from_github(TEST_CSV_URL)
    return anime_df, train_df, test_df

# Train models (simplified example)
def train_models(train_df, test_df):
    # Example model training code
    X = train_df.drop('rating', axis=1)
    y = train_df['rating']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    st.write(f"Model MSE: {mse}")
    
    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Predictions for test set
    test_pred = model.predict(test_df)
    return test_pred

# Main app
def main():
    st.title("Anime Recommendation System")
    
    tabs = ["Project Overview", "Loading Data", "Data Cleaning", "Exploratory Data Analysis", "Data Processing", "Model Training and Evaluation", "Model Improvement", "MLFlow Integration", "Conclusion", "Team Members", "Contact Us"]
    selected_tab = st.sidebar.selectbox("Select Tab", tabs)
    
    if selected_tab == "Project Overview":
        st.header("1. Project Overview")
        # Display the image
        image_url = "https://graphite-note.com/wp-content/uploads/2023/10/image-29.png"
        st.image(image_url, caption='Anime Recommendation System', use_column_width=True)
        
        st.subheader("1.1 Introduction")
        st.write("This project aims to build a recommendation system for anime titles...")
        
        st.subheader("1.2 Problem Statement")
        st.write("The problem is to predict user ratings for anime titles based on historical preferences...")
        
        st.subheader("1.3 Data Source")
        st.write("The data is sourced from myanimelist.net...")
        
        st.subheader("1.4 Aim")
        st.write("The aim is to create a collaborative and content-based recommender system...")
        
        st.subheader("1.5 Objectives")
        st.write("The objectives are to clean data, train models, and evaluate performance...")
    
    elif selected_tab == "Loading Data":
        st.header("2. Loading Data")
        st.subheader("2.1 Importing Packages")
        st.code("import pandas as pd\nimport numpy as np")
        
        st.subheader("2.2 Loading Data")
        anime_df, train_df, test_df = load_data()
        st.write("Data loaded successfully!")
        
        st.subheader("2.3 Verifying the Data")
        st.write("Anime DataFrame:")
        st.write(anime_df.head())
        st.write("Train DataFrame:")
        st.write(train_df.head())
        st.write("Test DataFrame:")
        st.write(test_df.head())

    elif selected_tab == "Data Cleaning":
        st.header("3. Data Cleaning")
        # Display the image
        image_url = "https://www.teraflow.ai/wp-content/uploads/2021/12/3-Big-Benefits-Of-Data-Cleansing-.jpg"
        st.image(image_url, caption='Data Cleaning', use_column_width=True)
        
        st.subheader("3.1 Importing Packages")
        st.code("import pandas as pd\nimport numpy as np")
        
        st.subheader("3.2 Cleaning")
        st.write("Cleaning data by handling missing values and duplicates...")
        # Data Cleaning
        anime_df, _, _ = load_data()
        anime_df.dropna(inplace=True)
        anime_df.drop_duplicates(inplace=True)
        st.write("Data after cleaning:")
        st.write(anime_df.head())

    elif selected_tab == "Exploratory Data Analysis":
        st.header("4. Exploratory Data Analysis")
        # Display the image
        image_url = "https://cdn.prod.website-files.com/63119622d2a6edf1d171e0bc/65d3a069871456bd33730869_GC2xT1oWgAA1rAC.jpeg"
        st.image(image_url, caption='Exploratory Data Analysis', use_column_width=True)
        
        st.subheader("4.1 Importing Packages")
        st.code("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns")
        
        st.subheader("4.2 Basic Statistics")
        anime_df, _, _ = load_data()
        st.write(anime_df.describe())
        
        st.subheader("4.3 Data Distribution")
        st.bar_chart(anime_df['genre'].value_counts())
        
        st.subheader("4.4 Correlation Analysis")
        corr_matrix = anime_df.corr()
        st.write(corr_matrix)
        
        st.subheader("4.5 Exploring Relationships")
        sns.pairplot(anime_df)
        st.pyplot()

    elif selected_tab == "Data Processing":
        st.header("5. Data Processing")
        # Display the image
        image_url = "https://www.managedoutsource.com/wp-content/uploads/2023/04/data-processing-methods.png"
        st.image(image_url, caption='Data Processing', use_column_width=True)
        
        st.subheader("5.1 Importing Packages")
        st.code("from sklearn.preprocessing import StandardScaler, LabelEncoder")
        
        st.subheader("5.2 Vectorization")
        st.write("Vectorizing text features...")
        # Example vectorization code
        
        st.subheader("5.3 Scaling")
        anime_df, _, _ = load_data()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(anime_df.select_dtypes(include=[np.number]))
        st.write("Scaled features:")
        st.write(scaled_features)
        
        st.subheader("5.4 Balancing")
        st.write("Handling imbalanced data...")
        # Example balancing code
        
        st.subheader("5.5 Encoding Categorical Variables")
        label_encoder = LabelEncoder()
        encoded_genre = label_encoder.fit_transform(anime_df['genre'])
        anime_df['encoded_genre'] = encoded_genre
        st.write("Encoded features:")
        st.write(anime_df.head())

    elif selected_tab == "Model Training and Evaluation":
        st.header("6. Model Training and Evaluation")
        # Display the image
        image_url = "https://analyticsindiamag.com/wp-content/uploads/2020/08/rohit-1.png"
        st.image(image_url, caption='Model Training and Evaluation', use_column_width=True)
        
        _, train_df, test_df = load_data()
        st.write("Training models...")
        predictions = train_models(train_df, test_df)
        st.write("Predictions on test set:")
        st.write(predictions)
    
    elif selected_tab == "Model Improvement":
        st.header("7.Model Improvement")
        # Display the image
        image_url = "https://neptune.ai/wp-content/uploads/blog/2022/Model%20improvement%20methods/model-improvement.jpg"
        st.image(image_url, caption='Model Improvement', use_column_width=True)
        
        st.subheader("7.1 Tuning Hyperparameters")
        st.write("Tuning model hyperparameters for better performance...")
        
        st.subheader("7.2 Ensemble Methods")
        st.write("Using ensemble methods to improve model accuracy...")
        
        st.subheader("7.3 Feature Engineering")
        st.write("Creating new features to enhance model performance...")
        
        st.subheader("7.4 Model Evaluation")
        st.write("Evaluating the model using cross-validation...")
    
    elif selected_tab == "MLFlow Integration":
        st.header("8. MLFlow Integration")
        # Display the image
        image_url = "https://content.techgig.com/photo/85350863/mlflow-1-14-0-launched-new-features-and-enhancements.jpg"
        st.image(image_url, caption='MLFlow Integration', use_column_width=True)
        
        st.subheader("8.1 Tracking Experiments")
        st.write("Tracking experiments with MLFlow...")
        
        st.subheader("8.2 Logging Parameters and Metrics")
        with mlflow.start_run():
            mlflow.log_param("param1", 5)
            mlflow.log_metric("metric1", 0.85)
            mlflow.sklearn.log_model(RandomForestRegressor(), "model")

    elif selected_tab == "Conclusion":
        st.header("9. Conclusion")
        # Display the image
        image_url = "https://assets.weforum.org/article/image/L15kdmKvZP4yLRop3qYY3kYExbovUgZ6IK2l7RkWQdY.jpg"
        st.image(image_url, caption='Conclusion', use_column_width=True)
        
        st.write("Summarizing the findings and results of the project...")
        st.write("This project successfully built and evaluated a recommendation system for anime titles, demonstrating the effectiveness of various data science techniques.")

    elif selected_tab == "Team Members":
        st.header("10. Team Members")
        st.subheader("10.1 Team Introduction")
        st.write("Meet the team behind this project...")
        st.write("Kentse Mphahlele") 
        st.write("Kamogelo Thalakgale")
        st.write("Neo Mbhele")
        st.write("Paballo Saku")
        st.write("Zakhele Mabuza-DEV")
        # Add more team members as needed
        
        st.subheader("10.2 Acknowledgements")
        st.write("Special thanks to everyone who contributed to this project.")

    elif selected_tab == "Contact Us":
        st.header("11. Contact Us")
        st.subheader("11.1 Contact Information")
        st.write("Feel free to reach out to us for any inquiries or collaborations...")
        st.write("Email: JB5@Explore.AI.co.za/Nzakhele15@gmail.com")
        st.write("Phone: +27 81 241 7465")
        # Add more contact details as needed
        
        st.subheader("11.2 Follow Us")
        st.write("Follow us on social media...")
        st.write("LinkedIn: [Link to LinkedIn](https://www.linkedin.com)")
        st.write("GitHub: [Link to GitHub](https://www.github.com)")

if __name__ == "__main__":
    main()
```

Save this code to a file named `app.py` and follow the steps mentioned above to run the Streamlit app. Let me know if you need any further assistance!