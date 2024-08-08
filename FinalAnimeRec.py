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
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

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
    
    if anime_df is None or train_df is None or test_df is None:
        st.stop()
    
    return anime_df, train_df, test_df

# Train models (simplified example)
def train_models(train_df):
    # Splitting the train data into training and validation sets
    X = train_df.drop('rating', axis=1)
    y = train_df['rating']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_val = model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    st.write(f"Validation MSE: {mse_val}")
    
    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model

# Content-Based Recommendation
def content_based_recommendation(anime_df, title, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    anime_df['description'] = anime_df['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(anime_df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(anime_df.index, index=anime_df['title']).drop_duplicates()
    if title not in indices:
        st.error(f"Title '{title}' not found in the dataset.")
        return []
    
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

    if user_id not in user_sim_df.index:
        st.error(f"User ID '{user_id}' not found in the dataset.")
        return []
    
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:top_n + 1].index
    similar_users_ratings = user_ratings.loc[similar_users]

    recommended_anime_ids = similar_users_ratings.mean(axis=0).sort_values(ascending=False).index[:top_n]
    recommended_anime = train_df[train_df['anime_id'].isin(recommended_anime_ids)]['title'].unique()

    return recommended_anime

# Load the pre-trained model (for collaborative filtering, if required)
def load_model():
    model_path = r"C:\Users\Zakhele\Downloads\ONE\anime_recommendation_model.pkl"
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

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
        st.pyplot(plt.gcf())  # Pass the figure object to st.pyplot()

    elif selected_tab == "Data Processing":
        st.header("5. Data Processing")
        image_url = "https://www.managedoutsource.com/wp-content/uploads/2023/04/data-processing-methods.png"
        st.image(image_url, caption='Data Processing', use_column_width=True)
        
        st.subheader("5.1 Importing Packages")
        st.code("from sklearn.preprocessing import StandardScaler, LabelEncoder")
        
        st.subheader("5.2 Vectorization")
        st.write("Vectorizing anime descriptions using TF-IDF...")
        anime_df, _, _ = load_data()
        vectorizer = TfidfVectorizer()
        anime_tfidf = vectorizer.fit_transform(anime_df['description'].fillna(''))
        st.write(f"Vectorized shape: {anime_tfidf.shape}")

    elif selected_tab == "Model Training and Evaluation":
        st.header("6. Model Training and Evaluation")
        image_url = "https://www.sparkbit.pl/wp-content/uploads/2020/11/Model-Evaluation-Visualization-1.png"
        st.image(image_url, caption='Model Training and Evaluation', use_column_width=True)
        
        st.subheader("6.1 Training Models")
        train_df, _, _ = load_data()[1:]
        model = train_models(train_df)
        
        st.subheader("6.2 Model Evaluation")
        st.write("Model trained and evaluated with the validation set.")

    elif selected_tab == "Model Improvement":
        st.header("7. Model Improvement")
        image_url = "https://turing.com/blog/wp-content/uploads/2022/12/Ensemble-Model.png"
        st.image(image_url, caption='Model Improvement', use_column_width=True)
        
        st.subheader("7.1 Hyperparameter Tuning")
        st.write("Tuning hyperparameters to improve the model...")

    elif selected_tab == "MLFlow Integration":
        st.header("8. MLFlow Integration")
        st.subheader("8.1 Experiment Tracking")
        st.write("Integrating MLFlow for experiment tracking and logging...")

    elif selected_tab == "Executable App":
        st.header("9. Executable App")
        image_url = "https://t3.ftcdn.net/jpg/05/67/70/64/360_F_567706405_8sCCK5zV9GpPApntqCGJlYIxOhQ9B4mS.jpg"
        st.image(image_url, caption='Executable App', use_column_width=True)
        
        st.subheader("9.1 Building Streamlit App")
        st.write("Building the app for real-time anime recommendations...")

    elif selected_tab == "Conclusion":
        st.header("10. Conclusion")
        image_url = "https://pm1.narvii.com/6688/cd556c45063087c139478dd492501113d0d6e6cc_hq.jpg"
        st.image(image_url, caption='Conclusion', use_column_width=True)
        
        st.subheader("10.1 Summary")
        st.write("Summarizing the findings and performance of the recommendation system...")

    elif selected_tab == "Team Members":
        st.header("11. Team Members")
        image_url = "https://cdn-icons-png.flaticon.com/512/492/492810.png"
        st.image(image_url, caption='Team Members', use_column_width=True)
        
        st.subheader("11.1 Meet the Team")
        st.write("Introducing the team members behind this project...")

    elif selected_tab == "Contact Us":
        st.header("12. Contact Us")
        image_url = "https://cdn-icons-png.flaticon.com/512/733/733547.png"
        st.image(image_url, caption='Contact Us', use_column_width=True)
        
        st.subheader("12.1 Get in Touch")
        st.write("For any inquiries or feedback, please reach out to us...")

if __name__ == "__main__":
    main()
