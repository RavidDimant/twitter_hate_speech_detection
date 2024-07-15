# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

st.set_page_config(page_title="Responsible AI Course", layout="centered")
# Centered text at the top of the page
st.markdown(
    """
    <div style="text-align: center; font-size:20px; margin-top: 0; padding-top: 0;">
        <b>Responsible AI, Law, Ethics & Society Course</b>
    </div>
    """,
    unsafe_allow_html=True
)

# creating page sections
site_header = st.container()
business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis = st.container()
feedbacks = st.container()
contact = st.container()

with site_header:
    st.title('Hate Speech Detection')
    st.subheader("By 'Legalytics Squad' - Group 4")
    # st.header("Responsible AI, Law, Ethics & Society Course")
    st.sidebar.image(Image.open('visualizations/GroupLogo.png'), width=250)
    st.write("""

    This project aims to **automate content moderation** to identify hate speech using **machine learning binary classification algorithms.** 

    The final model was a **Logistic Regression** model that used Count Vectorization for feature engineering. It produced an F1 of 0.3958 and Recall (TPR) of 0.624.  

    Check out the project repository [here](https://github.com/RavidDimant/twitter_hate_speech_detection/tree/master).
    Cloned the work by [Sidney Kung](https://www.sidneykung.com/)
    """)

with business_context:
    st.header('The Problem of Content Moderation')
    st.sidebar.markdown("[The Problem of Content Moderation](https://hatespeechdetection-fkbzq5w4mmyk9m3bgjdc6l-ravid"
                        ".streamlit.app/#the-problem-of-content-moderation)")
    st.write("""

    **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, uses human moderators to some extent, both domestically and overseas.

    Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.**  Usually, the difference between hate speech and offensive language comes down to subtle context or diction.

    """)

with data_desc:
    understanding, venn = st.columns(2)
    with understanding:
        st.text('')
        st.write("""
        The **data** for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.

        The `.csv` file has **24,802 rows** where **6% of the texts were labeled as "Hate Speech".**

        Each text's label was voted on by crowdsource and determined by majority rules.
        """)
    with venn:
        st.image(Image.open('visualizations/word_venn.png'), width=400)

with performance:
    description, conf_matrix = st.columns(2)
    with description:
        st.header('Final Model Performance')
        st.sidebar.markdown("[Final Model Performance](https://hatespeechdetection-fkbzq5w4mmyk9m3bgjdc6l-ravid"
                            ".streamlit.app/#final-model-performance)")
        st.write("""
        These scores are indicative of the two major roadblocks of the project:
        - The massive class imbalance of the dataset
        - The model's inability to identify what constitutes as hate speech
        """)
    with conf_matrix:
        st.image(Image.open('visualizations/normalized_log_reg_countvec_matrix.png'), width=400)

with tweet_input:
    st.header('Is Your Text Considered Hate Speech?')
    st.sidebar.markdown("[Try it yourself!](https://hatespeechdetection-fkbzq5w4mmyk9m3bgjdc6l-ravid.streamlit.app"
                        "/#is-your-text-considered-hate-speech)")
    st.write(
        """*Please note that this prediction is based on how the model was trained, so it may not be an accurate 
        representation.*""")
    # user input here
    user_text = st.text_input('Enter Text', max_chars=280)  # setting input as user_text

with model_results:
    if st.button("Submit"):
        st.subheader('Prediction:')
        if user_text:
            # processing user_text
            # removing punctuation
            user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
            # tokenizing
            stop_words = list(stopwords.words('english'))
            tokens = nltk.word_tokenize(user_text)
            # removing stop words
            stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
            # taking root word
            lemmatizer = WordNetLemmatizer()
            lemmatized_output = []
            for word in stopwords_removed:
                lemmatized_output.append(lemmatizer.lemmatize(word))

            # instantiating count vectorizor
            count = CountVectorizer(stop_words=stop_words)
            X_train = pickle.load(open('pickle/X_train_2.pkl', 'rb'))
            X_test = lemmatized_output
            X_train_count = count.fit_transform(X_train)
            X_test_count = count.transform(X_test)

            # loading in model
            final_model = pickle.load(open('pickle/final_log_reg_count_model.pkl', 'rb'))

            # apply model to make predictions
            prediction = final_model.predict(X_test_count[0])

            if prediction == 0:
                st.subheader('**Not Hate Speech**')
            else:
                st.subheader('**Hate Speech**')
            st.text('')

with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis with VADER')

        # explaining VADER
        st.write(
            """*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        # spacer
        st.text('')

        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)
        if sentiment_dict['compound'] >= 0.05:
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05:
            category = ("**Negative 🚫**")
        else:
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Text is rated as", category)
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**")
            st.write(sentiment_dict['neg'] * 100, "% Negative")
            st.write(sentiment_dict['neu'] * 100, "% Neutral")
            st.write(sentiment_dict['pos'] * 100, "% Positive")
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph)


with feedbacks:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.subheader("Feedback on Classification Results")
    st.sidebar.markdown("[Give us feedback](https://hatespeechdetection-fkbzq5w4mmyk9m3bgjdc6l-ravid.streamlit.app"
                        "/#feedback-on-classification-results)")
    st.write("Think our model was wrong in classifying your text? Let us know!")
    st.write("Write your text to be classified here and explain why you think it was wrong:")
    # Text area for user input
    user_text = st.text_area("Enter your text here")
    # Additional text area for user explanation
    user_explanation = st.text_area("Explain why you think the classification was wrong")
    # Button to submit feedback
    if st.button("Submit Feedback"):
        if user_text and user_explanation:
            st.success("Thank you for your feedback! We will review it ASAP.")
        else:
            st.warning("Please fill out both fields before submitting.")

st.write("")
st.write("")
st.write("")
st.write("")
st.markdown(
    """
    <div style="text-align: center;">
        Ravid Dimant | Alona Zafrir | Tal Shalom | Veronika Sorochenkova | Daniel Niazov
    </div>
    """,
    unsafe_allow_html=True
)
