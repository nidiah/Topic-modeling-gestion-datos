#!/usr/bin/env python

from os.path import isfile
from os import makedirs
import re
from random import sample
from json import dump, load
from nltk.tokenize import RegexpTokenizer
import spacy
from tqdm import tqdm
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis.gensim

AVAILABLE_LANGS = ['es', 'en']

SPACY_MODELS = {
    'es': 'es_core_news_lg',
    'en': 'en_core_web_lg'
}

### General
# Modelado de tópicos
def make_dictionary_and_matrix(documentos, filter_extremes=False):
    '''Makes gensim dictionary and doc_term_matrix objects for LDA model training.
    Returns (dictionary, doc_term_matrix)'''
    dictionary = corpora.Dictionary(documentos)
    if filter_extremes:
        dictionary.filter_extremes(no_below=2, no_above=0.02) ### filtra palabras exclusivas de un documento o de alta frecuencia. muchos docs
        #dictionary.filter_extremes(no_below=2, no_above=0.5) ### filtra palabras exclusivas de un documento o de alta frecuencia. pocos docs
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documentos]
    return (dictionary, doc_term_matrix)

def train_LDA_model(ntopics, dictionary, doc_term_matrix, output_path=None, seed=42):
    '''Receives topic number '''
    print(f"Training LDA model with {ntopics} topics")
    model = LdaModel(
        doc_term_matrix,
        num_topics=ntopics,
        id2word=dictionary,
        passes=20,
        eval_every=1,
        iterations=50,
        alpha='auto',
        eta='auto',
        random_state=seed,
    )
    if output_path:
        print(f"Saving in {output_path}")
        model.save(output_path)
    return model

def load_LDA_model(model_path):
    return LdaModel.load(model_path)

def make_model_path(models_dir, corpus_label, model_label, ntopics):
    model_path = f"{models_dir}/{corpus_label}-{model_label}-ntopics{ntopics:02}"
    return model_path

def train_several_LDA_models(documentos, topic_numbers_to_try, corpus_label, model_label, models_dir, overwrite=False, filter_extremes=False):
    models = {}
    
    makedirs(models_dir, exist_ok=True)

    for ntopics in topic_numbers_to_try:
        output_path = make_model_path(models_dir, corpus_label, model_label, ntopics)

        if not isfile(output_path) or overwrite:
            dictionary, doc_term_matrix = make_dictionary_and_matrix(documentos, filter_extremes)
            train_LDA_model(ntopics, dictionary, doc_term_matrix, output_path=output_path)
        else:
            print(f"Already trained for {ntopics} topics")

        models[ntopics] = load_LDA_model(output_path)
        
    return models

# Evaluación de modelos
def calculate_topic_coherence(models, docs, measures=["c_npmi", "c_uci", "u_mass", "c_v"], verbose=True, filter_extremes=False):
    '''models should be a dictionary of ntopics as keys, LDA models as values.
    Returns pandas.DataFrame of scores'''
    scores = []
    dictionary, doc_term_matrix = make_dictionary_and_matrix(docs, filter_extremes)
    
    models_iterator = models.items()
    if verbose:
        models_iterator = tqdm(models_iterator, desc='Calculating model coherence: ')
    
    for ntopics, model in models_iterator:
        scoring = {"ntopics": ntopics}
        for measure in measures:
            # Based on: https://radimrehurek.com/gensim/models/coherencemodel.html
            # The 'texts' must be the same preprocessing -if any- we used to generate the model we are loading!  
            cm = CoherenceModel(model=model, dictionary=dictionary, coherence=measure, texts=docs)
            scoring[measure] = cm.get_coherence()

        scores.append(scoring)
    
    return pd.DataFrame(scores)

def plot_cv(scores, corpus_label, model_label, models_dir, save=True):
    ax = scores.sort_values(by="ntopics").plot(x="ntopics", y="c_v", marker="o")
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score")
    ax.set_title(f'{corpus_label}-{model_label}')
    
    if save:
        plt.savefig(f'{models_dir}/{corpus_label}-{model_label}-coherence.png')

    plt.show()
    
# Visualización de tópicos
def plot_LDA_topics(model, docs, output_path=None, show=True, notebook=True, filter_extremes=False):
    dictionary, doc_term_matrix = make_dictionary_and_matrix(docs, filter_extremes)
    
    if notebook:
        pyLDAvis.enable_notebook()
        
    data = pyLDAvis.gensim.prepare(model, doc_term_matrix, dictionary)
        
    if output_path:
        print(f"Saving topic visualization to: {output_path}")
        pyLDAvis.save_html(data, output_path)
    
    if show:
        return data
    
def topic_table(topic, model_label, corpus_label):
    model_name = f'{corpus_label}-{model_label}'
    df = pd.DataFrame(topic, columns=["word", "word_proba"])
    df["model_name"] = model_name
    df = df.reset_index().rename(columns={"index":"word_ranking"})
    df["word_ranking"] = df["word_ranking"] + 1
    return df

def make_table_all_topics(best_model, model_label, corpus_label):
    topics_best_model = []
    for i in range(best_model.num_topics):
        topic = best_model.show_topic(i, topn=15)
        df = topic_table(topic, model_label, corpus_label)
        df["topic_id"] = i + 1
        topics_best_model.append(df)

    tabla = pd.concat(topics_best_model, ignore_index=True)
    tabla = tabla[['model_name', 'topic_id', 'word_ranking', 'word', 'word_proba']]
    return tabla


### Informes financieros
def create_collection(cont):
    '''Crea una colección de textos a partir de txt de informes financieros taggeados por fecha'''
    cont = re.sub(r'»', '', cont)
    cont = re.sub(r'\n\n', '', cont)
    cont = re.sub(r'<soustit=[0-9]{2}>\n', '', cont)
    textos = re.split(r'<[^>]+>', cont)
    textos = [texto for texto in textos if len(texto) > 1] # elimina items vacíos
    print(f"La colección tiene {len(textos)} textos")
    return textos

def join_named_entities(spacy_texts):
    modified_texts = []
    for spacy_text in tqdm(spacy_texts):
        modified_text = str(spacy_text)

        for entity in spacy_text.ents:
            if '_' not in str(entity): # No procesar los bigramas
                entity_words = str(entity).split()
                entity_merged = "_".join(entity_words)
                modified_text = modified_text.replace(str(entity), entity_merged)

        modified_texts.append(modified_text.split())
    return modified_texts

### Narrativas covid
def infer_lang_from_corpus_label(label):
    # Example label: dhcovid_2020-7-11_2020-7-17_es_mx
    return label.split("_")[-2]

def read_tweets(path):
    '''Reads a txt from https://covid.dh.miami.edu/get/ and returns a list of tweets.'''
    with open(path, 'r') as fi:
        data = fi.read()
        data = set(data.split('\n') ) # elimina duplicados
    return list(data)

def read_stopwords(extra_stopwords_path):
    with open(extra_stopwords_path, 'r') as fi:
        extra_stopwords = fi.read()
        extra_stopwords = extra_stopwords.split('\n')
    return extra_stopwords

def remove_emojis(tweet):
    return re.sub(r':[^: ]+?:', '', tweet)

def remove_stopwords(lang, tweets, extra_stopwords_path):
    if lang == 'es':
        stop_words = stopwords.words('spanish')
    else:
        stop_words = stopwords.words('english')
    stop_words_extra = read_stopwords(extra_stopwords_path)
    stop_words = stop_words+stop_words_extra
    clean_tweets = [[token for token in texto if token not in stop_words] for texto in tweets]
    return clean_tweets

def count_words_in_tweet(tweet):
    return len(tweet.split())

def remove_one_word_tweets(tweets):
    return [tw for tw in tweets if count_words_in_tweet(tw) > 1]

def dump_processed_tweets_as_json(tweets, filepath):
    """tweets should be a list of lists of strings."""
    with open(filepath, "w") as f:
        dump(tweets, f)

def pipeline(path, extra_stopwords_path, NE_path, sample_n=None, overwrite=False, filter_extremes=False):
    # Assumes these previous steps:
    # $ python -m spacy download en_core_web_lg
    # $ python -m spacy download es_core_news_lg
    
    corpus_label = path.replace('.txt', '')
    processed_tweets_path = path.replace(".txt", ".processed-tweets.json")

    if not isfile(processed_tweets_path):
        lang = infer_lang_from_corpus_label(corpus_label)
        tweets = read_tweets(path)

        if sample_n and len(tweets) > sample_n:
            tweets = sample(tweets, min(sample_n, len(tweets)))

        print(f"[{corpus_label}] Preprocessing")
        tweets_noemojis = [remove_emojis(tweet).strip() for tweet in tweets]
        tweets_noemoji = [tweet for tweet in tweets_noemoji if tweet]
        
        clean_tweets = remove_stopwords(lang, tweets_noemoji)
        
        print(f"[{corpus_label}] Make bigrams")
        bigram = Phrases(clean_tweets, min_count=15)
        tweets_bigrams = clean_tweets.copy()
        for idx in tqdm(range(len(clean_tweets))):
            for token in bigram[clean_tweets[idx]]:
                if '_' in token:
                    tweets_bigrams[idx].append(token)
                    
        if not isfile(NE_path):
            print(f"[{corpus_label}] Identify Named Entities")
            spacy_nlp = spacy.load(SPACY_MODELS[lang])
            spacy_tweets = [spacy_nlp(tweet, disable=["tagger", "parser"]) for tweet in tqdm(tweets_bigrams)]
            with open(f'{corpus_label}_NE.lst', 'w') as fi:
                for tweet in spacy_tweets:
                    for entity in tweet.ents:
                        entity_words = str(entity).split()
                        if len(entity_words) > 1:
                            fi.write(f'{" ".join(entity_words)}\n')
            print(f"Dump Named Entities in {NE_path} for revision")
            ner_tweets = join_named_entities(spacy_tweets)
        else:
            NE_list = read_tweets(NE_path)
            print(f"[{corpus_label}] Join Named Entities")
            tweets_ner = []
            for tweet in tqdm(tweets_bigrams):
                tweet = ' '.join(tweet)
                for entity in NE_list:
                    entity_merged = '_'.join(entity.split())
                    tweet = tweet.replace(entity, entity_merged)
                tweets_ner.append(tweet.split())
        
        print(f"[{corpus_label}] Dump processed tweets as JSON")
        dump_processed_tweets_as_json(ner_tweets, processed_tweets_path)
    else:
        print(f"[{corpus_label}] Load processed tweets from JSON")
        ner_tweets = load_processed_tweets_from_json(processed_tweets_path)
    
    print(f"[{corpus_label}] Unsupervised learning")
    model_label = '2gram_ner_LDA'
    models_dir = f".{corpus_label}_models"

    models = train_several_LDA_models(
        tweets=ner_tweets,
        topic_numbers_to_try=range(3, 51),
        corpus_label=corpus_label,
        model_label=model_label,
        models_dir=models_dir,
        overwrite=overwrite,
        filter_extremes=filter_extremes
    )