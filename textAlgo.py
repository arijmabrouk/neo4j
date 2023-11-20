import os
import PyPDF2
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy

nltk.download('stopwords')
# Function to preprocess the document
def preprocess(document):
    # Download stopwords for English and French if not already downloaded
   

    # Load stopwords for English and French
    stop_words_english = stopwords.words('english')
    stop_words_french = stopwords.words('french')

    # Combine English and French stopwords
    stop_words = stop_words_english + stop_words_french

    processed_doc = [
        word.lower() for word in simple_preprocess(document, deacc=True)
        if word.lower() not in stop_words and word.lower() not in string.punctuation
    ]
    return processed_doc

# Function to create a corpus from a list of documents
def create_corpus(docs):
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return dictionary, corpus

# Function to extract text from a PDF file
def extract_text_from_pdf(filepath):
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ''
        for page in range(num_pages):
            page_obj = pdf_reader.pages[page]
            text += page_obj.extract_text()
        return text



def getTopicForOneLDA(pdf_filepath):
    document_text = extract_text_from_pdf(pdf_filepath)

    # Preprocess the document
    processed_doc = preprocess(document_text)

    # Create a corpus
    dictionary, corpus = create_corpus([processed_doc])

    # Apply LDA
    num_topics = 5  # Adjust the number of topics as needed
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda_model.print_topics(num_topics=num_topics)

    json_data = {}
    for topic_id, topic_info in topics:
        topic_words = [entry.split('"')[1] for entry in topic_info.split(' + ')]
        topic_data = [{"percentage": float(entry.split('*')[0]), "keyword": entry.split('"')[1]} for entry in topic_info.split(' + ')]
        json_data[str(topic_id)] = topic_data

    # Convert the JSON data to a list of topics
    topics_list = list(json_data.values())

    return topics_list


def getTopicForOneHLDA(pdf_filepath):
    pass


def getAllTopicsLDA(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            topics = getTopicForOneLDA(file_path)

            # Convert the topics list to a list of objects with "topics" key
            
            results.append(topics)

    return results

def NMF_topicsByOne(file_path, num_topics):
    text = extract_text_from_pdf(file_path)
    processed_doc = preprocess(text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(processed_doc)])

    nmf_model = NMF(n_components=num_topics, init='random', random_state=42)
    nmf_model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(top_words)
        topic_str = ' '.join(str(val) for val in topic)
        print(f"Topic {topic_idx + 1}: {topic_str}")
    return topics



def NMF_topicsByFolder(folder_path, num_topics):
    topics_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            topics = NMF_topicsByOne(file_path, num_topics)
            topics_list.append(topics)

    return topics_list


def build_graph(sentences):
    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Build graph
    graph = nx.from_numpy_array(similarity_matrix)
    return graph



def getGraphBasedSummarize_pg(file_path):
    extracted_text = extract_text_from_pdf(file_path)
    sentences = extracted_text.split('. ')
    graph = build_graph(sentences)  # Build the graph using the sentences

    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for _, s in ranked_sentences[:5]]
    summary_list = [sentence.strip() for sentence in summary_sentences]

    return summary_list



def getGraphBasedSummarize_tr(file_path):
    extracted_text = extract_text_from_pdf(file_path)
    parser = PlaintextParser.from_string(extracted_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summarizer.stop_words = stopwords.words('english') + stopwords.words('french')  # Add custom stop words as needed
    print(stopwords.words('english') + stopwords.words('french'))
    summary_list = []
    summary = summarizer(parser.document, 5)  # Summarize the document with 5 sentences

    for sentence in summary:
        summary_list.append(str(sentence))

    return summary_list
    

def generate_text_summary(filepath, max_sentences=8):
    input_text = extract_text_from_pdf(filepath)
    
    def preprocess_text(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return doc
    
    def extract_phrases(tree):
        phrases = set()  # Use a set to remove duplicates
        for sent in tree.sents:
            phrases.add(sent.text)
        return list(phrases)  # Convert the set back to a list
    
    def generate_summary(phrases, max_sentences):
        summary = '. '.join(phrases[:max_sentences])
        return summary
    
    tree = preprocess_text(input_text)
    phrases = extract_phrases(tree)
    summary_list = phrases[:max_sentences]  # Get the first 'max_sentences' from the list
    
    return summary_list

def getTreeBasedSummarize_pg(file_path):
    summary_list = generate_text_summary(file_path, max_sentences=8)  # Adjust the number of sentences as desired
    print(summary_list)
    return summary_list
