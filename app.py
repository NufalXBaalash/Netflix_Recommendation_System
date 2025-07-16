import streamlit as st
import faiss
import numpy as np
import pandas as pd
import google.generativeai as genai
import os
import requests
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie recommendations based on your preferences!")

# Sidebar for API key input
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    help="Get your API key from Google AI Studio"
)

# Function to download file from Google Drive
def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive using the file ID"""
    try:
        # Create the direct download URL
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return False

# Function to ensure files are downloaded
def ensure_files_downloaded():
    """Download required files if they don't exist"""
    files_to_download = [
        {
            'filename': 'index',
            'file_id': '1p2dwABn9vtJo-3Jr4p3vp0yAI58vC62T',
            'description': 'FAISS index'
        },
        {
            'filename': 'netflix_titles.csv',
            'file_id': '11xhBTYRcUyVjTskxqqLzXw-v9b8MNaHu',
            'description': 'Netflix titles dataset'
        }
    ]
    
    for file_info in files_to_download:
        if not os.path.exists(file_info['filename']):
            st.info(f"Downloading {file_info['description']}...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Downloading {file_info['filename']}...")
            
            success = download_from_google_drive(file_info['file_id'], file_info['filename'])
            
            if success:
                progress_bar.progress(1.0)
                status_text.text(f"✅ {file_info['description']} downloaded successfully!")
                st.success(f"{file_info['description']} ready!")
            else:
                st.error(f"Failed to download {file_info['description']}")
                return False
    
    return True
# Load the FAISS index and movie data
@st.cache_resource
def load_index_and_data():
    try:
        # Load FAISS index
        index = faiss.read_index('index')
        
        # Load Netflix titles dataset
        df = pd.read_csv('netflix_titles.csv')
        
        return index, df
    except Exception as e:
        st.error(f"Error loading index or data: {str(e)}")
        return None, None

# Function to get embedding from Gemini
def get_embedding(text, api_key):
    try:
        genai.configure(api_key=api_key)
        
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        
        embedding = np.array([response["embedding"]], dtype="float32")
        return embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

# Function to get movie recommendations
def get_recommendations(user_input, index, df, api_key, top_k=5):
    # Get embedding for user input
    embedding = get_embedding(user_input, api_key)
    
    if embedding is None:
        return None
    
    # Search for similar movies
    D, I = index.search(embedding, top_k)
    
    # Get the best matches
    indices = I.flatten()
    distances = D.flatten()
    
    # Create recommendation data
    recommendations = []
    for idx, distance in zip(indices, distances):
        if idx < len(df):
            movie_data = df.iloc[idx]
            
            # Handle Netflix dataset columns
            title = movie_data.get('title', f'Title {idx}')
            description = movie_data.get('description', 'No description available')
            
            # Additional Netflix-specific info
            movie_type = movie_data.get('type', 'Unknown')
            release_year = movie_data.get('release_year', 'Unknown')
            rating = movie_data.get('rating', 'Not Rated')
            duration = movie_data.get('duration', 'Unknown')
            genre = movie_data.get('listed_in', 'Unknown')
            
            recommendations.append({
                'index': idx,
                'title': title,
                'description': description,
                'type': movie_type,
                'release_year': release_year,
                'rating': rating,
                'duration': duration,
                'genre': genre,
                'similarity': 1 - distance,
                'distance': distance
            })
    
    return recommendations

# Main application
def main():
    # Ensure required files are downloaded
    if not ensure_files_downloaded():
        st.error("Failed to download required files. Please check your internet connection and try again.")
        st.stop()
    
    # Load index and data
    index, df = load_index_and_data()
    
    if index is None or df is None:
        st.stop()
    
    # User input section
    st.header("Tell us what you're looking for:")
    user_input = st.text_area(
        "Describe your movie preferences:",
        placeholder="e.g., I want to watch a romantic comedy with great acting and witty dialogue...",
        height=100
    )
    
    # Recommendation button
    if st.button("Get Recommendations", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        elif not user_input.strip():
            st.error("Please enter your movie preferences.")
        else:
            with st.spinner("Finding perfect movies for you..."):
                recommendations = get_recommendations(user_input, index, df, api_key)
                
                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"Found {len(recommendations)} movie recommendations for you!")
                    
                    # Display recommendations in an organized way
                    st.header("🎬 Your Movie Recommendations")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns([3, 1])
                    
                    for idx, movie in enumerate(recommendations, 1):
                        # Create a container for each movie
                        with st.container():
                            # Movie header with type badge
                            movie_type_color = "🎬" if movie['type'] == 'Movie' else "📺"
                            st.markdown(f"### {idx}. {movie_type_color} {movie['title']}")
                            
                            # Movie badges
                            badge_col1, badge_col2, badge_col3, badge_col4 = st.columns(4)
                            with badge_col1:
                                st.markdown(f"**Type:** {movie['type']}")
                            with badge_col2:
                                st.markdown(f"**Year:** {movie['release_year']}")
                            with badge_col3:
                                st.markdown(f"**Rating:** {movie['rating']}")
                            with badge_col4:
                                st.markdown(f"**Duration:** {movie['duration']}")
                            
                            # Create columns for movie info and similarity score
                            movie_col1, movie_col2 = st.columns([4, 1])
                            
                            with movie_col1:
                                st.markdown(f"**Description:**")
                                st.write(movie['description'])
                                
                                # Genre information
                                st.markdown(f"**Genres:** {movie['genre']}")
                            
                            with movie_col2:
                                # Similarity score with color coding
                                similarity_percent = movie['similarity'] * 100
                                if similarity_percent >= 80:
                                    color = "green"
                                elif similarity_percent >= 60:
                                    color = "orange"
                                else:
                                    color = "red"
                                
                                st.markdown(f"**Similarity**")
                                st.markdown(f"<span style='color:{color}; font-size:20px; font-weight:bold'>{similarity_percent:.1f}%</span>", unsafe_allow_html=True)
                                
                                # Progress bar for similarity
                                st.progress(float(movie['similarity']))
                            
                            # Add some spacing between movies
                            st.markdown("---")
                    
                    # Summary statistics
                    st.subheader("📊 Recommendation Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_similarity = sum(movie['similarity'] for movie in recommendations) / len(recommendations)
                        st.metric("Average Similarity", f"{avg_similarity*100:.1f}%")
                    
                    with col2:
                        best_match = max(recommendations, key=lambda x: x['similarity'])
                        st.metric("Best Match", best_match['title'])
                    
                    with col3:
                        movie_count = sum(1 for movie in recommendations if movie['type'] == 'Movie')
                        st.metric("Movies", movie_count)
                    
                    with col4:
                        tv_count = sum(1 for movie in recommendations if movie['type'] == 'TV Show')
                        st.metric("TV Shows", tv_count)
                        
                else:
                    st.error("Failed to get recommendations. Please check your API key and try again.")

# Additional features
with st.sidebar:
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This app uses:
    - **FAISS** for similarity search
    - **Google Gemini** for embeddings
    - **Streamlit** for the interface
    - **Netflix dataset** with 8,000+ titles
    - **Auto-download** from Google Drive
    """)
    
    st.subheader("How to use:")
    st.markdown("""
    1. Enter your Gemini API key
    2. Wait for files to download (first time only)
    3. Describe your movie preferences
    4. Get personalized Netflix recommendations!
    """)

    st.subheader("Features:")
    st.markdown("""
    - **Smart Recommendations**: AI-powered content matching
    - **Netflix Integration**: Real titles and descriptions
    - **Similarity Scores**: Color-coded match percentages
    - **Auto-setup**: Downloads required files automatically
    """)

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")