# 👗 Fashion Recommender System

## 📌 Overview
This project is a Content-Based Fashion Recommender System that suggests visually similar fashion products based on image embeddings.

The system extracts deep features using a pre-trained CNN model and recommends similar items using cosine similarity.

## 🛠 Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras

 ## ⚙️ How It Works

1. Extract image embeddings using ResNet50.
2. Store embeddings in a pickle file.
3. Compute cosine similarity.
4. Recommend top 5 similar fashion items.

## 📂 Project Structure

├── app.py
├── main.py
├── test.py
├── embedding.pkl (Not uploaded due to size)
├── filenames.pkl (Not uploaded due to size)
├── images/ (Dataset not uploaded)


## 🚀 Installation

1.Install dependencies
   pip install -r requirements.txt

2. Run the app
   streamlit run app.py


## 🚀 Future Improvements

- Add collaborative filtering
- Deploy on AWS / Render
- Improve UI design
- Add user login system

## 👩‍💻 Author

Anjali Kumari  
B.Tech - Computer Science  
Rungta College of Engineering and Technology
   
