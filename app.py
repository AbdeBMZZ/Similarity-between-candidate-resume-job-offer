from collections import namedtuple
import pandas as pd 
from flask import Flask, jsonify
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

Inv = namedtuple('candidate', 'name, email, address, github, phone, skills, SIMILARITY_SCORE')

app = Flask(__name__)

@app.route('/')
def index():
    # reading the resume
    with pdfplumber.open("cv.pdf") as pdf:
        page = pdf.pages[0]
        text = page.extract_text()
    
    # extracting the data
    name = re.search(r'(?<=\n)[A-Z]{1}[a-z]{1,} [A-Z]{1}[a-z]{1,}', text).group(0)
    email = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text).group(0)
    address = re.search(r'[a-zA-Z0-9]{1,3} [a-zA-Z0-9 ]{1,}', text).group(0)
    github = re.search(r'github.com/[a-zA-Z0-9-]+', text).group(0)
    phone = re.search(r'[0-9]{2} [0-9]{2} [0-9]{2} [0-9]{2} [0-9]{2}', text).group(0)
    skills = re.findall(r'[A-Z]{1}[a-z]{1,}', text)


    data = []
    

    required_skills = ['python', 'django', 'flask', 'html', 'css', 'javascript', 'react', 'conception']


    txt1 = ' '.join(required_skills)
    txt2 = ' '.join(skills)

    processed_txt1 = ' '.join([word for word in txt1.split() if word not in stopwords.words('english')])
    processed_txt2 = ' '.join([word for word in txt2.split() if word not in stopwords.words('english')])

    corpus = [processed_txt1, processed_txt2]
    vectorizer = TfidfVectorizer()
    
    tfidf = vectorizer.fit_transform(corpus)

    similarity_matrix = cosine_similarity(tfidf)[0,1]
    

    data.append(Inv(name, email, address, github, phone, skills, similarity_matrix))

    df = pd.DataFrame(data, columns=['name', 'email', 'address', 'github', 'phone', 'skills', 'SIMILARITY_SCORE'])
    df.to_csv(name + '.csv', index=False)
    return jsonify(df.to_dict())


if __name__ == '__main__':
    app.run(debug=True)