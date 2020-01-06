# %% [markdown]
# # Website Text Mining

# %%
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import requests
import justext
from bs4 import BeautifulSoup, SoupStrainer
import json
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import os
import subprocess
from neo4j import GraphDatabase
from pathlib import Path
import streamlit as st
import re

# %%
# Declaring title, info and load spacy models
SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
DEFAULT_TEXT = "www.clear.ai"
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


st.sidebar.title("Interactive Web Scraping")
st.sidebar.markdown(
    """
Process text scraped from a website and visualize named entities and key phrases. NER uses spaCy's built-in
[displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
"""
)

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()


# %% [markdown]
# Firstly, we create a NamedEntityRecognition class. Each object of this class will represent a given webpage and specifically hold the information regarding the named entities present within the webpage

# %%
class NamedEntityRecognition:
    
    def __init__(self, url):
        
        """
        Args: url of the webpage as a string
        
        Just instantiates all the lists of named entities
        
        """
        
        self.url = url
        self.companies = []
        self.people = []
        self.keywords = []
        self.locations = []
        self.events = []
        self.products = []
        self.text = None
        self.doc = None
        
    def visualise(self, jupyter=True):
        
        """
        Args: None
        
        Returns a html rendered visualisation of the named entities within the homepage of the url using displacy
        
        """
        
        if str(self.doc) != "":
            return displacy.render(self.doc, style="ent", jupyter=jupyter)
        else:
            return "No entities found."
    
    def jsonify(self):
        
        """
        Args: None
        
        Returns a json version of all the named entities within the webpage
        """
        
        list_of_entities = [self.companies, self.people, self.keywords, self.locations, self.events, self.products]
        list_of_entity_types = ["companies", "people", "keywords", "locations", "events", "products"]
        zip_object = zip(list_of_entity_types, list_of_entities)
        entity_dict = dict(zip_object)
        
        return json.dumps(entity_dict, indent = 4, ensure_ascii=False)
    
    def get_people_wiki_pages(self):
        
        """
        Args: None
        
        Returns a dictionary consisting of the people identified within the webpage as the key, whilst the value attempts
        to return the url and first line of the article for the closest match to the person on wikipedia
        
        *** STILL NEEDS WORK, RETURNS UNEXPECTED RESULTS IF THE PERSON IS NOT ON WIKIPEDIA ***
        
        """
        
        wiki = []
        
        for person in self.people:
            try:
                result = wikipedia.page(person)
                summary = wikipedia.summary(person, sentences=1)
                wiki.append([result.url, summary])
            except:
                wiki.append(["",""])
        zip_object = zip(self.people, wiki)
        
        return dict(zip_object)

    def __get_links(self):
        
        """
        Args: None
        
        Private function which attempts to extract all the relevant links from the original webpage that point to other
        pages on the same website and returns the urls in a list
        
        """
    
        new_url = "http://" + self.url
        html=requests.get(new_url)
        links = []

        for link in BeautifulSoup(html.text, parse_only=SoupStrainer('a')):
            try:
                new_link = (link['href'])

                short_url = self.url[4:]
                if (new_link.find(short_url) >= 0) and (new_link.count('@') == 0):
                    idx=new_link.find(short_url) + len(short_url)
                    end=new_link.find('/',idx+1)
                    if end == -1:
                        end = len(new_link)
                    links.append(new_link[:end])

                elif (new_link.count('/', 1) <= 1) and (str(new_link)[:4] != "http") and (new_link.count('#') == 0) and (new_link.count('@') == 0):

                    if (new_link.count('/', 1) == 1) and (new_link[:-1] != "/"):
                        continue
                    links.append(self.url + new_link)
            except:
                continue

        return list(set(links))
    
    def get_salience(self):
        
        """
        Args: None
        
        Returns a measure of the relative importance of all the named entities. It attempts to do this by simply counting the
        frequency of all the terms both on the webpage and on all the other relevant pages to which the original page links 
        using the __get_links() method. The function then returns a dictionary in descending order of count.
        
        """
        
        corpus=[self.text]
        links = self.__get_links()

        for link in links:

            if (link[:4] != "http"):
                link = "http://" + link

            try:
                html = requests.get(link)
            except:
                html=""
            
            text = ""
            if str(html) == "<Response [200]>":
                
                try:
                    paragraphs = justext.justext(html.text.encode('utf-8'), justext.get_stoplist("English"))
                    for paragraph in paragraphs:
                        if not paragraph.is_boilerplate:
                            text+= paragraph.text + "\n"
                except:
                    print("Could not parse text")
            corpus.append(text)

        all_entities = self.companies + self.people + self.keywords + self.locations + self.events + self.products
        counts = []
        
        for entity in all_entities:
            counts.append(sum(entity in s for s in corpus))
        
        entity_dict = dict(zip(all_entities, counts))
        
        return {k: v for k, v in sorted(entity_dict.items(), key=lambda x: x[1],reverse=True)}

# %% [markdown]
# Now, we create an object which parses a file containing a list of urls and extracts content from the webpages, including named entities, storing them in instances of the NamedEntityRecognition class.

# %%
class Extract:
    
    def __init__(self, csv_or_url):
        
        """
        Args: filename containing a list of url strings (with 'url' header) or an individual url string
        
        Takes a list of urls or single url and initialises some variables. 
        Adds a function removing whitespace to the spacy pipeline if it has not already been added
        
        """
        
        my_file = Path(csv_or_url)
        
        if my_file.is_file():
            sources=pd.read_csv(csv_or_url,header=None,names=["url"])
        else:
            sources = pd.DataFrame({'url':csv_or_url}, index=[0])
        
        self.sources = sources
        self.bad_urls = []
        self.extracted = self.__extract_text()
        self.ner_list = None
        
        if "__remove_whitespace_entities" not in (dict(nlp.pipeline).keys()):
            nlp.add_pipe(self.__remove_whitespace_entities, after='ner')

    def __extract_text(self):
        
        """
        Args: None
        
        Private function which is called on initiliatisation of the object. Loops through every url in the list and extracts
        non-boilerplate content before saving it as list. Keeps track of urls which are either broken or timeout before
        returning the webpage and saves it in the data member "bad_urls"
        
        """
        
        extracted=[]
        for url in self.sources['url']:

            if ("http://" not in url):
                if ("https://" not in url): 
                    url = "http://"+url
            
            try:
                html = requests.get(url)
            except requests.exceptions.RequestException as e:
                self.bad_urls.append(url)
                html = None
                print (e)
            
            if str(html) == "<Response [200]>":
                
                text = ""
                paragraphs = justext.justext(html.text.encode('utf-8'), justext.get_stoplist("English"))
                for paragraph in paragraphs:
                    if not paragraph.is_boilerplate:
                        text+= paragraph.text + "\n"
            else:
                text="<ERROR>"

            extracted.append(text)

        return extracted
    
    def __remove_whitespace_entities(self, doc):
        
        """
        Args: string representing an individual document
        
        A function added to the spacy pipeline due to a bug in spacy which results in certain whitespace characters being 
        recognised as entities: https://github.com/explosion/spaCy/issues/2870
        
        """
        
        doc.ents = [e for e in doc.ents if not e.text.isspace()]
        return doc
        
    def extract_entities(self):
        
        """
        Args: None
        
        Uses the spacy namer entity recogniser to extract named entities from the non-boilerplate content of each webpage.
        Additionally uses the Rake algorithm to extract keywords from the same non-boilerplate content. Creates an instance of
        the NamedEntityRecognition class for each webpage and stores the addresses to the objects in a list called "ner_list"
        
        """
        
        ner_list=[]
        r = Rake()
        k = 10 #top number of key words to extract

        for i, url in enumerate(self.sources['url']):
            ner_list.append(NamedEntityRecognition(url))
            
            ner_list[i].text = self.extracted[i]
            r.extract_keywords_from_text(ner_list[i].text)
            top_k_keywords = [value for value in dict(r.get_ranked_phrases_with_scores()[:k]).values()]
            ner_list[i].keywords = top_k_keywords
            
            doc = nlp(self.extracted[i])
            obj = ner_list[i]
            obj.doc = doc
            
            for entity in doc.ents:
                if (entity.label_ == "ORG"):
                    if entity.text not in  obj.companies: obj.companies.append(entity.text)
                elif (entity.label_ == "PERSON"):
                    if entity.text not in  obj.people: obj.people.append(entity.text)
                elif (entity.label_ == "GPE"):
                    if entity.text not in  obj.locations: obj.locations.append(entity.text)            
                elif (entity.label_ == "EVENT"):
                    if entity.text not in  obj.events: obj.events.append(entity.text)        
                elif (entity.label_ == "PRODUCT"):
                    if entity.text not in  obj.products: obj.products.append(entity.text)
        
        self.ner_list = ner_list
        return
        
    def get_number_of_urls(self):
        
        """
        Args: None
        
        Simple function to just return the number of urls in the input file
        
        """
        return len(self.sources)

# %% [markdown]
# Finally, we create a class for adding our extracted entities and keywords to a neo4j graph:

# %%
class Neo4j:
    
    def __init__(self, NER_object):
        """
        Args: an object of type NamedEntityRecognition
        
        Takes an object of type NamedEntityRecognition and extracts the url and JSON from the object. 
        Also connects to a neo4j database.
        
        """
        self.uri = "bolt://localhost:7687"
        self.driver = GraphDatabase.driver(self.uri, auth=("neo4j", "4jNeo"))
        self.json = json.loads(NER_object.jsonify())
        self.url = NER_object.url
        self.salience = NER_object.get_salience()
        
    def __create_query(self, tx, test):
        """
        Args: a neo4j driver session
        
        Internal function which creates and runs the query that extracts the information from the JSON and
        inserts them as nodes into the neo4j graph.
        """
        temp_id = 1
        query = ("CREATE (name:website {url:'"+self.url+"'})")
        for i in self.json:
            for j in self.json[i]:
                query += " CREATE (node" + str(temp_id) + ":" + i + " {name:'"+j+"', salience:" + str(self.salience[j]) +"})"
                query += " CREATE (node" + str(temp_id) + ")-[:EXTRACTED_FROM]->(name)"
                temp_id += 1
        if test is False: tx.run(query)
        
        return ("Nodes added to graph:" + str(temp_id))
                
    def add_nodes(self, test=False):
        """
        Args: none
        
        External function which calls __create_function from a neo4j driver session
        """
        with self.driver.session() as session:
            msg = session.read_transaction(self.__create_query, test)

        return msg
# %%
# Now let's call everything using streamlit

st.header("Enter URL Below")
text = st.text_input("URL", DEFAULT_TEXT)
model_load_state = st.info(f"Analysing...")
webpages = Extract(text)
webpages.extract_entities()
model_load_state.empty()
extracted_data = webpages.ner_list[0]

if st.button("Extract"):

    # Displacy NER Section
    st.subheader('Visualise Named Entities')
    html = extracted_data.visualise(jupyter=False)
    html = re.sub(r'\n\s*\n', '\n\n', html)
    html = html.replace("\n\n", "\n")
    st.write(html,unsafe_allow_html=True)

    # View JSON
    st.subheader('JSON')
    st.json(extracted_data.jsonify())

    # View Salience
    st.subheader('Relative Importance')
    salience = extracted_data.get_salience()
    vals = [i for i in salience.values()]
    norm = [format(float(i)/max(vals),'1.2f') for i in vals]
    df = pd.DataFrame({'Extracted': [i for i in salience.keys()],'Relative Importance': norm})
    st.dataframe(df)

# Export to neo4j
st.subheader('Export to neo4j')
if st.button("Export"):
    if extracted_data is None:
        st.write("Please extract the data first")
    else:
        n4j=Neo4j(extracted_data)
        msg = n4j.add_nodes()
        st.write(msg)
        st.balloons()