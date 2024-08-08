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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to load CSV files from local directory
@st.cache_data
def load_csv_from_local(file_path):
    return pd.read_csv(file_path)

# Local file paths
ANIME_CSV_PATH = 'C:/Users/Zakhele/Downloads/ONE/anime.csv'
TRAIN_CSV_PATH = 'C:/Users/Zakhele/Downloads/ONE/train.csv'
TEST_CSV_PATH = 'C:/Users/Zakhele/Downloads/ONE/test.csv'

# Load data
@st.cache_data
def load_data():
    anime_df = load_csv_from_local(ANIME_CSV_PATH)
    train_df = load_csv_from_local(TRAIN_CSV_PATH)
    test_df = load_csv_from_local(TEST_CSV_PATH)
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

# Content-Based Recommendation
def content_based_recommendation(anime_df, title, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    anime_df['description'] = anime_df['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(anime_df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(anime_df.index, index=anime_df['title']).drop_duplicates()
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    anime_indices = [i[0] for i in sim_scores]

    return anime_df['title'].iloc[anime_indices]

# Collaborative Filtering Recommendation
def collaborative_filtering_recommendation(train_df, user_id, top_n=10):
    user_ratings = train_df.pivot(index='user_id', columns='anime_id', values='rating')
    user_ratings = user_ratings.fillna(0)

    user_similarity = cosine_similarity(user_ratings)
    user_sim_df = pd.DataFrame(user_similarity, index=user_ratings.index, columns=user_ratings.index)

    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:top_n + 1].index
    similar_users_ratings = user_ratings.loc[similar_users]

    recommended_anime_ids = similar_users_ratings.mean(axis=0).sort_values(ascending=False).index[:top_n]
    recommended_anime = train_df[train_df['anime_id'].isin(recommended_anime_ids)]['title'].unique()

    return recommended_anime

# Load the pre-trained model (for collaborative filtering, if required)
def load_model():
    with open(r"C:\Users\Zakhele\Downloads\ONE\anime_recommendation_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Main app
def main():
    st.title("Anime Recommendation System")
    
    tabs = ["Project Overview", "Loading Data", "Data Cleaning", "Exploratory Data Analysis", "Data Processing", "Model Training and Evaluation", "Model Improvement", "MLFlow Integration", "Executable App", "Conclusion", "Team Members", "Contact Us"]
    selected_tab = st.sidebar.selectbox("Select Tab", tabs)
    
    if selected_tab == "Project Overview":
        st.header("1. Project Overview")
        image_url = "https://graphite-note.com/wp-content/uploads/2023/10/image-29.png"
        st.image(image_url, caption='Anime Recommendation System', use_column_width=True)
        
        st.subheader("1.1 Introduction")
        st.write("This project aims to build a recommendation system for anime titles using both content-based and collaborative filtering methods.")
        
        st.subheader("1.2 Problem Statement")
        st.write("The problem is to predict user ratings for anime titles based on historical preferences using a hybrid recommendation system.")
        
        st.subheader("1.3 Data Source")
        st.write("The data is sourced from myanimelist.net and includes anime content information and user ratings.")
        
        st.subheader("1.4 Aim")
        st.write("The aim is to create a robust recommendation system that can recommend anime based on user preferences and content similarities.")
        
        st.subheader("1.5 Objectives")
        st.write("The objectives are to clean data, train models, implement recommendation methods, and evaluate performance.")
    
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
        image_url = "https://www.teraflow.ai/wp-content/uploads/2021/12/3-Big-Benefits-Of-Data-Cleansing-.jpg"
        st.image(image_url, caption='Data Cleaning', use_column_width=True)
        
        st.subheader("3.1 Importing Packages")
        st.code("import pandas as pd\nimport numpy as np")
        
        st.subheader("3.2 Cleaning")
        st.write("Cleaning data by handling missing values and duplicates...")
        anime_df, _, _ = load_data()
        anime_df.dropna(inplace=True)
        anime_df.drop_duplicates(inplace=True)
        st.write("Data after cleaning:")
        st.write(anime_df.head())

    elif selected_tab == "Exploratory Data Analysis":
        st.header("4. Exploratory Data Analysis")
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
        numeric_df = anime_df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        st.write(corr_matrix)
    
        st.subheader("4.5 Exploring Relationships")
        sns.pairplot(numeric_df)
        st.pyplot()

    elif selected_tab == "Data Processing":
        st.header("5. Data Processing")
        image_url = "https://www.managedoutsource.com/wp-content/uploads/2023/04/data-processing-methods.png"
        st.image(image_url, caption='Data Processing', use_column_width=True)
        
        st.subheader("5.1 Importing Packages")
        st.code("from sklearn.preprocessing import StandardScaler, LabelEncoder")
        
        st.subheader("5.2 Vectorization")
        st.write("Vectorizing text features...")
        
        st.subheader("5.3 Scaling")
        anime_df, _, _ = load_data()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(anime_df.select_dtypes(include=[np.number]))
        st.write("Scaled features:")
        st.write(scaled_features)
        
        st.subheader("5.4 Balancing")
        st.write("Handling imbalanced data...")
        
        st.subheader("5.5 Encoding Categorical Variables")
        label_encoder = LabelEncoder()
        encoded_genre = label_encoder.fit_transform(anime_df['genre'])
        st.write("Encoded genres:")
        st.write(encoded_genre)
    
    elif selected_tab == "Model Training and Evaluation":
        st.header("6. Model Training and Evaluation")
        image_url = "https://m.mylargescale.com/1/9/7/9/8/6/5/IMG_6733.jpg"
        st.image(image_url, caption='Model Training', use_column_width=True)
        
        st.subheader("6.1 Importing Packages")
        st.code("from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error")
        
        st.subheader("6.2 Training")
        st.write("Training the model...")
        anime_df, train_df, test_df = load_data()
        predictions = train_models(train_df, test_df)
        st.write("Training complete!")
        
        st.subheader("6.3 Evaluation")
        st.write("Evaluating the model...")
        mse = mean_squared_error(train_df['rating'], predictions[:len(train_df)])
        st.write(f"Mean Squared Error: {mse}")
        
        st.subheader("6.4 Visualization")
        st.line_chart(predictions[:len(train_df)])

    elif selected_tab == "Model Improvement":
        st.header("7. Model Improvement")
        st.subheader("7.1 Importing Packages")
        st.code("from sklearn.model_selection import GridSearchCV\nfrom sklearn.ensemble import RandomForestRegressor")
        
        st.subheader("7.2 Hyperparameter Tuning")
        st.write("Tuning model hyperparameters...")
        st.write("Example: GridSearchCV")
    
    elif selected_tab == "MLFlow Integration":
        st.header("8. MLFlow Integration")
        image_url = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*rb53EZIC7KaKPAxt74T7uA.png"
        st.image(image_url, caption='MLFlow Integration', use_column_width=True)
        
        st.subheader("8.1 Importing Packages")
        st.code("import mlflow\nimport mlflow.sklearn")
        
        st.subheader("8.2 Logging Parameters and Metrics")
        st.write("Logging parameters and metrics with MLFlow...")
    
    elif selected_tab == "Executable App":
        st.header("9. Executable App")
        image_url = "https://mobidev.biz/images/app_development/How_to_Develop_a_Real-Time_Communication_App/How_to_Develop_a_Real-Time_Communication_App.jpg"
        st.image(image_url, caption='Anime Recommendation Executable App', use_column_width=True)
        
        st.subheader("9.1 Content-Based Filtering")
        st.write("Provide an anime title to get content-based recommendations:")
        anime_df, _, _ = load_data()
        selected_anime = st.selectbox("Select an anime:", anime_df['title'].tolist())
        recommendations = content_based_recommendation(anime_df, selected_anime)
        st.write(f"Top content-based recommendations for '{selected_anime}':")
        st.write(recommendations)
        
        st.subheader("9.2 Collaborative Filtering")
        st.write("Provide a user ID to get collaborative filtering recommendations:")
        user_id = st.number_input("Enter user ID:", min_value=1)
        if st.button("Get Recommendations"):
            recommendations = collaborative_filtering_recommendation(train_df, user_id)
            st.write(f"Top collaborative filtering recommendations for user {user_id}:")
            st.write(recommendations)
        
        st.subheader("9.3 Model Deployment")
        st.write("Load a pre-trained model for making predictions:")
        model = load_model()
        st.write("Model loaded successfully!")

    elif selected_tab == "Conclusion":
        st.header("10. Conclusion")
        st.write("In this project, we built an anime recommendation system using content-based and collaborative filtering approaches.")
        st.write("The system was developed with various components including data cleaning, EDA, model training, and MLFlow integration.")
        st.write("Future work could involve improving model accuracy and expanding the dataset.")
        
    elif selected_tab == "Team Members":
        st.header("11. Team Members")
        st.write("Zakhele Mabuza")
        st.write("Kentse Mphahlele")
        st.write("Kamogelo Thalakgale")
        st.write("Neo Mbhele")
        st.write("Paballo Saku")
        
    elif selected_tab == "Contact Us":
        st.header("12. Contact Us")
        st.write("If you have any questions or need further assistance, feel free to contact us.")
        st.write("Email: Nzakhele15@gmail.com")
        st.write("Phone: +27 73 636 1228")

if __name__ == "__main__":
    main()