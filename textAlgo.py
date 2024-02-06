import os
import PyPDF2
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaModel,HdpModel
from nltk.corpus import stopwords
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
import urllib

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
import openai

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
    # Extract text from PDF
    document_text = extract_text_from_pdf(pdf_filepath)

    # Check if document text is empty or None
    if not document_text:
        print("Error: Document text is empty.")
        return []

    # Preprocess the document
    processed_doc = preprocess(document_text)

    # Check if processed document is empty or None
    if not processed_doc:
        print("Error: Processed document is empty.")
        return []

    # Create a corpus
    dictionary, corpus = create_corpus([processed_doc])

    # Check if the dictionary or corpus is empty
    if not dictionary or not corpus:
        print("Error: Dictionary or corpus is empty.")
        return []

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


def get_topic_for_one_hdp(pdf_filepath):
    # Extract text from PDF
    document_text = extract_text_from_pdf(pdf_filepath)

    # Check if document text is empty or None
    if not document_text:
        print("Error: Document text is empty.")
        return []

    # Preprocess the document
    processed_doc = preprocess(document_text)

    # Check if processed document is empty or None
    if not processed_doc:
        print("Error: Processed document is empty.")
        return []

    # Create a corpus
    dictionary, corpus = create_corpus([processed_doc])

    # Check if the dictionary or corpus is empty
    if not dictionary or not corpus:
        print("Error: Dictionary or corpus is empty.")
        return []

    # Apply HDP
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)

    # Get topics
    topics = hdp_model.show_topics()

    json_data = {}
    for topic_id, topic_info in topics:
        try:
            # Check the length of topic_info before unpacking
            if len(topic_info) == 2:
                word, weight = topic_info
            elif len(topic_info) == 1:
                word = topic_info[0]
                weight = 1.0  # Set a default weight if not available
            else:
                raise ValueError("Unexpected format for topic_info")

            topic_data = [{"percentage": weight, "keyword": word}]
            json_data[str(topic_id)] = topic_data
        except Exception as e:
            print(f"Error processing topic {topic_id}: {e}")
            

    # Convert the JSON data to a list of topics
    topics_list = list(json_data.values())
    print(f"topics_list: {topics_list}")
    return topics_list


def getSummerizeAi(pdf_filepath):
    openai.api_key = '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'
    
    # Replace this with your actual PDF text extraction logic
    document_text = extract_text_from_pdf(pdf_filepath)

    # Provide the extracted document text as the prompt for summarization
    prompt = f"Summarize the following PDF document:\n\n{document_text}"

    # Make a request to the OpenAI API
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150  # You can adjust this parameter to control the length of the summary
    )

    # Get the generated summary
    summary = response['choices'][0]['text']

    return summary



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

    # Add a check for empty documents
    if not text.strip():
        print(f"Warning: Document '{file_path}' is empty or contains only stop words.")
        return []

    processed_doc = preprocess(text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(processed_doc)])

    nmf_model = NMF(n_components=num_topics, init='random', random_state=42)
    nmf_model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic in nmf_model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.extend(top_words)  # Extend the list instead of appending a sublist

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
def extract_keywords_from_sentence(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Extracting nouns as keywords, you can adjust this based on your requirements
    keywords = [token.text.lower() for token in doc if token.pos_ == "NOUN"]

    return keywords


def getGraphBasedSummarize_pg(file_path):
    extracted_text = extract_text_from_pdf(file_path)
    sentences = extracted_text.split('. ')

    # Build the graph using the sentences
    graph = build_graph(sentences)

    # Calculate PageRank scores
    scores = nx.pagerank(graph)

    # Sort sentences by PageRank score in descending order
    ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Extract top 5 sentences and their scores
    top_sentences = [{"sentence": sentences[idx].strip(), "score": score} for idx, score in ranked_sentences[:5]]

    # Extract keywords from the top sentences
    keywords = []
    for sentence_info in top_sentences:
        sentence = sentence_info["sentence"]
        score = sentence_info["score"]
        sentence_keywords = extract_keywords_from_sentence(sentence)  # Replace with your own function
        keywords.extend([{"keyword": keyword, "score": score} for keyword in sentence_keywords])

    return keywords




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
    
def extract_keywords_from_sentence_tree(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Extracting nouns and adjectives as keywords, filtering out non-alphabetic characters
    keywords = [token.lemma_.lower() for token in doc if token.is_alpha and token.pos_ in ["NOUN", "ADJ"]]

    return keywords
def generate_text_summary(filepath, max_sentences=8):
    input_text = extract_text_from_pdf(filepath)
    
    def preprocess_text(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return doc
    
    def extract_keywords(tree):
        keywords = set()  # Use a set to remove duplicates
        for sent in tree.sents:
            keywords.update(extract_keywords_from_sentence_tree(sent.text))
        return list(keywords)  # Convert the set back to a list
    
    def generate_summary(keywords, max_sentences):
        summary = '. '.join(keywords[:max_sentences])
        return summary
    
    tree = preprocess_text(input_text)

    keywords = extract_keywords(tree)
    print("summmmmm",keywords)

    
     
    return keywords

def getTreeBasedSummarize_pg(file_path):
    summary = generate_text_summary(file_path, max_sentences=8)  # Adjust the number of sentences as desired
    # print("summmmmm",summary)
    return summary
