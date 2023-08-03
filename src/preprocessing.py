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
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)

    # Replace Positive ratings
    text = re.sub(r'\b(100|[6-9][0-9])%\b', 'RATING_SCORE_POS', text)
    text = re.sub(r'\b([6-9]|10)/10\b', 'RATING_SCORE_POS', text)
    text = re.sub(r'\b([4-5])/5\b', 'RATING_SCORE_POS', text)

    # Replace Neutral ratings
    text = re.sub(r'\b(4[1-9]|5[0-9])%\b', 'RATING_SCORE_NEUTRAL', text)
    text = re.sub(r'\b5/10\b', 'RATING_SCORE_NEUTRAL', text)
    text = re.sub(r'\b3/5\b', 'RATING_SCORE_NEUTRAL', text)

    # Replace Negative ratings
    text = re.sub(r'\b([0-9]|[0-3][0-9]|40)%\b', 'RATING_SCORE_NEG', text)
    text = re.sub(r'\b[0-4]/10\b', 'RATING_SCORE_NEG', text)
    text = re.sub(r'\b[0-2]/5\b', 'RATING_SCORE_NEG', text)

    # Remove remaining numbers
    text = re.sub(r'\d+', '', text)

    # Expand contractions
    text = contractions.fix(text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize and Lemmatize
    word_tokens = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]

    # Remove stopwords (and any lingering 's')
    filtered_text = [word for word in lemmatized_words if word not in stop_words and word != 's']
    text = ' '.join(filtered_text)

    # Remove punctuation
    replace_with_space = '.,;:!?-'
    remove_punctuation = ''.join([punc for punc in string.punctuation if punc not in replace_with_space])
    translation_table = str.maketrans(replace_with_space, ' ' * len(replace_with_space), remove_punctuation)

    text = text.translate(translation_table)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text
