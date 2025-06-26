# 🕵️‍♂️ Fake News Detector

<i>Because the internet needed **another** fake news detector—but this one’s *actually good*.</i>

Welcome to the **Fake News Detector**, an overly concerned piece of Python code trying to help humanity tell facts from fiction. Whether you're tired of reading about miracle weight loss mangoes or yet another "Elon buys XYZ" headline, this machine learning project will gladly rain on their parade.

---

## 🚀 Features

- Text preprocessing that's cleaner than your browser history.
- **Dual-model detection**: Support Vector Machine (_aka that nerd who always gets it right_) and Passive Aggressive Classifier (_the rebellious cousin who thrives on confrontation_).
- Data visualization for all you graph lovers.
- Confusion matrices that look exactly like how you feel reading fake news.
- Custom article predictions to let you play detective.

---

## 🧠 Behind the Scenes

This project uses:

- **TF-IDF Vectorizer** to turn words into meaningful numbers.
- **Passive Aggressive Classifier** for its "I don't care but I actually do" approach.
- **SVM** because, well, it works. Period.
- Preprocessing includes stemming, stopword removal, regex cleansing, and sarcasm filters (ok, not really).

---

# 📦 Requirements
Install dependencies using:
```
pip install -r requirements.txt
```
Or manually, if you like to suffer.

Download the Dataset from my drive:
```
https://drive.google.com/file/d/1wvQvvvDos0EJYv1A3-tPApZdzHACUm6X/view?usp=sharing
```

---

## 🛠 How to Run

1. Clone this amazing repository.
2. Drop your dataset into the root folder (we expect columns named `text` and `label`).
3. Adjust the `DATA_PATH` in `fake_news_detector.py` if you like living dangerously.
4. Run the Python script:

```bash
python fake_news_detector.py
```

Sit back and enjoy as the program judges your articles more critically than your relatives at a wedding.

---


# 📊 Sample Output
```
=== Fake News Detection System ===
Loading dataset...
Training models...
Evaluating performance...
Plotting confusion matrix...
Regretting reading the news...
```

---
# 🧪 Test Example
Input:
"BREAKING: Scientists say chocolate is the new kale!"

Output:
PAC Prediction: FAKE<br>
SVM Prediction: FAKE<br>
Consensus: FAKE<br>
Confidence: 98.76% – So yeah, nice try.<br>

---
## 📊 Model Performance

The system uses two complementary models:

- **Passive Aggressive Classifier**: Excellent for online learning and large datasets
- **Support Vector Machine**: Robust linear classifier for text data

Both models are evaluated using:
- Accuracy Score
- F1 Score  
- Confusion Matrix
- Classification Report
---

## 🔍 How It Works

1. **Data Preprocessing**:
   - Remove HTML tags, URLs, and punctuation
   - Tokenization and normalization
   - Stopword removal and stemming
   - TF-IDF vectorization

2. **Model Training**:
   - Train both PAC and SVM models
   - 70/30 train-test split with stratification

3. **Prediction**:
   - Process input article through preprocessing pipeline
   - Generate predictions from both models
   - Provide consensus prediction with confidence metrics

---

## 🎓 Educational Value

This project demonstrates:
- End-to-end ML pipeline development
- NLP preprocessing techniques
- Model evaluation and comparison
- Social impact AI applications
- Clean, documented code practices
---

## 🔮 Future Enhancements

- Integration with pre-trained embeddings (BERT, Word2Vec)
- Real-time news scraping and classification
- API endpoint development
- Enhanced feature engineering with metadata
- Deep learning models (LSTM, Transformer)
---

## 📈 Success Metrics

- Achieve >85% accuracy on test dataset
- Clear documentation and reproducible results
- Professional presentation for portfolio showcase
- Practical application for social good

---

# 👨‍💻 Author
Made with coffee, code, and a sprinkle of existential dread by Devansh Singh.

Feel free to connect with me on dksdevansh@gmail.com if you’re into cool projects, sarcastic readmes, or you just want to say hi.

---

# 📜 License
You can do whatever you want with this code. Just don’t make it tell people that pineapple belongs on pizza. That would be crossing the line.

---

# ⚠️ Disclaimer
No fake news was emotionally harmed during the making of this project. But a few poorly written headlines were judged. Harshly.