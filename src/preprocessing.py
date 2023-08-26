import re
import string
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = remove_html(text)
    text = replace_ratings(text)
    text = remove_numbers(text)
    text = expand_contractions(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    return text


def preprocess_text_RNN(text):
    text = remove_html(text)
    text = expand_contractions(text)
    text = text.lower()
    text = remove_whitespace(text)
    return text


def remove_html(text):
    return re.sub('<.*?>', '', text)


def replace_ratings(text):
    # Replace Positive ratings
    text = re.sub(r'\b(100|[6-9][0-9])%\b', 'RATING_SCORE_POS', text)
    text = re.sub(r'\b([6-9]|10)/10\b', 'RATING_SCORE_POS', text)
    text = re.sub(r'\b([4-5])/5\b', 'RATING_SCORE_POS', text)

    # Replace Neutral ratings
    text = re.sub(r'\b(4[1-9]|5[0-9])%\b', '', text)
    text = re.sub(r'\b5/10\b', '', text)
    text = re.sub(r'\b3/5\b', '', text)

    # Replace Negative ratings
    text = re.sub(r'\b([0-9]|[0-3][0-9]|40)%\b', 'RATING_SCORE_NEG', text)
    text = re.sub(r'\b[0-4]/10\b', 'RATING_SCORE_NEG', text)
    text = re.sub(r'\b[0-2]/5\b', 'RATING_SCORE_NEG', text)
    return text


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def expand_contractions(text):
    return contractions.fix(text)


def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
    filtered_text = [word for word in lemmatized_words if word not in stop_words and word != 's']
    return ' '.join(filtered_text)


def remove_punctuation(text):
    replace_with_space = '.,;:!?-'
    remove_punc = ''.join([punc for punc in string.punctuation if punc not in replace_with_space])
    translation_table = str.maketrans(replace_with_space, ' ' * len(replace_with_space), remove_punc)
    return text.translate(translation_table)


def remove_whitespace(text):
    return ' '.join(text.split())
