from urllib.parse import urlparse # for feature extraction

from sklearn.naive_bayes import BernoulliNB # naiiiive bayes model
from sklearn.model_selection import train_test_split # separates dataset into training data and testing data
from sklearn.metrics import classification_report # creates a report on how well the model has done

# for the crf training
import sklearn_crfsuite
from sklearn_crfsuite import metrics

import pandas as pd # interprets csv file data
import pickle # serializing objects (turning data types into formats they can be stored in e.g. bytes)

from training_data import crf_training_data # imports the training data for the crf (too long so it looked ugly here)

Domain_Keywords = ["manga", "manhwa", "manhua", "comic", "scan", "comics", "scans"]
Category_keywords = ["manga", "manhwa", "manhua", "comics", "read", "series"]
Chapter_Keywords = ["chapter", "chap", "ch", "ep", "episode"]

def bayes_feature_extraction(url: str):
    url = url.lower()

    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    segments = [s for s in path.split("/") if s]
    
    
    has_domain_keyword = int(any(k in domain for k in Domain_Keywords))
    has_category_keyword = int(any(k in segments for k in Category_keywords))
    has_chapter_keyword = int(any(k in segments for k in Chapter_Keywords))

    # has japanese slug

    has_chap_num = int(len(segments) > 0 and segments[-1].isdigit())
    has_chap_next_to_num = int(has_chap_num and segments[len(segments) - 2] in Chapter_Keywords) # make this smarter

    # add checks for manga specific websites - e.g. '.xyz' '.to' 'manga-name', 'chap-num in dif location', /en/

    return [
        has_domain_keyword,
        has_category_keyword,
        has_chapter_keyword,
        has_chap_num,
        has_chap_next_to_num
    ]

def train_naive_bayes(url_csv: str):
    df = pd.read_csv(url_csv, sep=';') # stores csv as a dataframe(df) object
    
    urls = df["url"].tolist()
    labels = df["label"].tolist()
    
    X =  [bayes_feature_extraction(url) for url in urls]
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = BernoulliNB()
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    with open("manga_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

def looks_like_title_part(token):
    return (
        token.isalpha() and
        token not in Category_keywords and
        token not in Chapter_Keywords
    )

def token_features(sequence, index: int):
    token = sequence[index]

    features = {
        "token": token,

        "is_digit": token.isdigit(),
        "is_alpha": token.isalpha(),
        "is_short_number": token.isdigit() and int(token) < 1000,

        "is_structural_word": token in ["manga", "manhwa", "manhua", "read", "series", "comics", "title"],
        "is_chapter_indicator": token in Chapter_Keywords,

        "prev_token": sequence[index -1] if index > 0 else "START",
        "next_token": sequence[index +1] if index < len(sequence) + 1 else "END",

        "prev_title_candidate": looks_like_title_part(sequence[index-1]) if index > 0 else False,
        "next_title_candidate": looks_like_title_part(sequence[index+1]) if index < len(sequence)-1 else False,

        "prev_is_structural": sequence[index-1] in Category_keywords if index > 0 else False,
        "prev_is_chapter_indicator": sequence[index - 1] in Chapter_Keywords if index > 0 else False,

        "next_chapter_indicator": any(t in Chapter_Keywords for t in sequence[index:]),

        "distance_to_chapter": next((i for i, t in enumerate(sequence[index:]) if t in Chapter_Keywords), -1),

        "position": index,
        "is_last": index == len(sequence) - 1,
        "is_first": index == 0,
    }

    return features

def sequence_feature_extraction(sequence):
    # cycles labels every token in sequence
    return [token_features(sequence, i) for i in range(len(sequence))]



# used to actually run and test the AI model on individual instances of URLs
def isManga():
    url = input("Enter URL to test here: ")
    with open("manga_classifier.pkl", "rb") as f:  # "rb" = read binary
        model = pickle.load(f)

    features = bayes_feature_extraction(url)
    prediction = model.predict([features])

    if prediction:
        print(f'{url} IS a manga website! It will automatically be added to the database.')
    else:
        print(f'{url} is NOT a manga website. It will not be automatically added to the database.')

if True:
    isManga()
