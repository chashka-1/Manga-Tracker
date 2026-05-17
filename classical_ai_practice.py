from urllib.parse import urlparse # for feature extraction

from sklearn.naive_bayes import BernoulliNB # naiiiive bayes model
from sklearn.model_selection import train_test_split # separates dataset into training data and testing data
from sklearn.metrics import classification_report # creates a report on how well the model has done
import pandas as pd # interprets csv file data
import pickle # serializing objects (turning data types into formats they can be stored in e.g. bytes)


#### FEATURE EXTRACTION FUNCTION for training

Domain_Keywords = ["manga", "manhwa", "manhua", "comic", "scan"]
Category_keywords = ["manga", "manhwa", "manhua", "comics", "read"]
Chapter_Keywords = ["chapter", "/chap/", "/ch/", "/ep/", "episode"]

def feature_extraction(url: str):
    url = url.lower()

    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    segments = [s for s in path.split("/") if s]

    has_domain_keyword = int(any(k in domain for k in Domain_Keywords))
    has_category_keyword = int(any(k in segments for k in Category_keywords))
    has_chapter_keyword = int(any(k in segments for k in Chapter_Keywords))

    print(url)
    print(parsed_url)
    print(domain)
    print(path)
    print(segments)

test: str = "https://asurascans.com/comics/magic-academys-genius-blinker-030ff47a/chapter/96"
feature_extraction(test)

# lowercase url
# separate URL into segments
# search through list and vectorify
#    domain-name: has manga/manhwa/comic(s)/scan(s)/
#    has title/long path 
#    has "read"
#    has "chapter/chap/ch/ep/episode"
#    has number after chap section
#    has numeric section @ very end


