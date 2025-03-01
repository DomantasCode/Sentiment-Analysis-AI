import pandas as pd
import random
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Atsisiunčiame filmų apžvalgų korpusą (jei dar neatsisiųsta)
nltk.download('movie_reviews')

# Įkelkite ir paruoškite duomenų rinkinį iš NLTK filmų apžvalgų
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# Atsitiktinai sumaišykite dokumentus, kad teigiamų ir neigiamų atsiliepimų seka būtų išmaišyta
random.shuffle(documents)

# Sukurkite pandas DataFrame su tekstu ir nuotaikos etiketėmis
df = pd.DataFrame(documents, columns=['text', 'sentiment'])

# Padalinkite duomenų rinkinį į mokymo ir testavimo dalis
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# Sukurkite duomenų srautą, kuris vektorizuoja tekstą ir treniruoja klasifikatorių
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# Treniruokite klasifikatorių
pipeline.fit(X_train, y_train)

# Nuspėkite nuotaiką testavimo rinkiniui
y_pred = pipeline.predict(X_test)

# Įvertinkite modelio veikimą
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
