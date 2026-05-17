from urllib.parse import urlparse # for feature extraction

from sklearn.naive_bayes import BernoulliNB # naiiiive bayes model
from sklearn.model_selection import train_test_split # separates dataset into training data and testing data
from sklearn.metrics import classification_report # creates a report on how well the model has done
import pandas as pd # interprets csv file data
import pickle # serializing objects (turning data types into formats they can be stored in e.g. bytes)

Domain_Keywords = ["manga", "manhwa", "manhua", "comic", "scan", "comics", "scans"]
Category_keywords = ["manga", "manhwa", "manhua", "comics", "read", "series"]
Chapter_Keywords = ["chapter", "chap", "ch", "ep", "episode"]

def feature_extraction(url: str):
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
    
    X =  [feature_extraction(url) for url in urls]
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = BernoulliNB()
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    with open("manga_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

# with open("manga_classifier.pkl", "rb") as f:  # "rb" = read binary
#    model = pickle.load(f)

training_csv = "training_urls.csv"
train_naive_bayes(training_csv)