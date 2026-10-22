# 🎬 YouTube Growth Predictor

A machine learning web app that predicts a YouTube channel's **growth tier** (Rising, Established or Dominant Creator) from its statistics, with an interactive Streamlit analytics dashboard.

🔗 **Live app:** _add your Streamlit link here_

## ✨ Features
- Predicts a channel's growth tier using a **Random Forest Classifier** (200 estimators)
- Shows confidence scores for each tier
- **15 engineered features**, including views per subscriber, channel age, upload rate and 30-day growth rates
- Analytics dashboard: top channels, categories, average earnings and countries
- Trained on the **Global YouTube Statistics 2023** dataset (Kaggle, 1,006 channels)

## 🛠️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn (Random Forest Classifier), Joblib
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **UI / Deployment:** Streamlit, Streamlit Community Cloud

## 🚀 Run Locally
```bash
git clone https://github.com/likhith17d-blip/youtube-growth-predictor.git
cd youtube-growth-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
├── Global_YouTube_Statistics.csv   # Dataset
├── youtube_growth_model.pkl        # Trained Random Forest model
├── le_category.pkl                 # Label encoders
├── le_channel_type.pkl
└── le_country.pkl
```

## 👥 Team
Built as a team project for **CSE23602 – Machine Learning** at **Global Academy of Technology, Bengaluru**, under the guidance of Prof. Yashaswini K.

- Prasad KS
- RS Giri
- **Likhith D** – [LinkedIn](https://www.linkedin.com/in/dlikhith17)
- Manoj DA
