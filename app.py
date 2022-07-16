import pandas as pd
import numpy as np
import streamlit as st
import joblib
from PIL import Image
import pickle
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import string

import time

import plotly.graph_objects as go
import plotly.express as px


# Using streamlit plotly to create UI for the Sentiment Analysis Model to be used



@st.cache(allow_output_mutation=True)

#create a function to load the model

def load(vectoriser_path, model_path):

    #load vectoriser
    file = open(vectoriser_path, 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # load model
    file = open(model_path, 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel

# create a function to return a dataframe with model sentiment predictions based on input data
def inference(vectoriser, model, tweets, cols):

    text = tweets.split(";")
    finaldata = []
    textdata = vectoriser.transform(lemmatize_process(preprocess(text)))
    sentiment = model.predict(textdata)

    sentiment_prob = model.predict_proba(textdata)
    for index,tweet in enumerate(text):
        if sentiment[index] == 1:
            sentiment_probFinal = sentiment_prob[index][1]

        else:
            sentiment_probFinal = sentiment_prob[index][0]

        sentiment_probFinal2 = "{}%".format(round(sentiment_probFinal*100,2))
        finaldata.append((tweet, sentiment[index], sentiment_probFinal2))
    #convert list into pd df
    df = pd.DataFrame(finaldata, columns = ['Text', 'Sentiment', 'Probability (Confidence Interval)'])
    df = df.replace([0,1], ["Negative", "Positive"])

    return df



# defining method to return 2nd parameter for lemmatization that is POS tag

def get_wordnet_pos_tag(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# create a method to perform lemmatization with POS tags identified via a pos_tag method

def lemmatize_process(preprocessedtext):
    lemma = WordNetLemmatizer()
    
    finalprocessedtext = []
    for tweet in preprocessedtext:
        text_pos = pos_tag(word_tokenize(tweet))
        words = [x[0] for x in text_pos]
        pos = [x[1] for x in text_pos]
        tweet_lemma = " ".join([lemma.lemmatize(a,get_wordnet_pos_tag(b)) for a,b in zip(words,pos)])
        finalprocessedtext.append(tweet_lemma)
    return finalprocessedtext

# excel input

def predict_Excel(vectoriser, model, text):
    finaldata = []
    
    textdata = vectoriser.transform(lemmatize_process(preprocess(text)))
    sentiment = model.predict(textdata)
    
    # print(model.classes_)
    sentiment_prob = model.predict_proba(textdata)
    
    for index, tweet in enumerate(text):
        sentiment_probFinal = sentiment_prob[index][0]
        
        sentiment_probFinal2 = "{}".format(round(sentiment_probFinal,4))
        finaldata.append((tweet, sentiment[index], sentiment_probFinal2))
        
    #Convert the list into a pandas df
    
    df = pd.DataFrame(finaldata, columns = ['Text', 'Sentiment', 'Probability'])
    df = df.replace([0,1], ["Negative", "Positive"])
    return df

#defined method with preprocessing functions

def preprocess(textdata):
    
    #defining dictionary containing all emojis with their meanings
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
             ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
             ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\\\': 'annoyed', 
             ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
             '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
             '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
             ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
    
    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                   'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                   'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                   'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
                   'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                   'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                   'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                   'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                   'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                   's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                   't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                   'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
                   'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                   'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                   'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                   "youve", 'your', 'yours', 'yourself', 'yourselves']
    
    processedText = []
    
  
    # Defining regex patterns.\n",
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
 
    userPattern = '@[^\\s]+'

    #alphaPattern      = "[^a-zA-Z]
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        #replace all URLs with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        
        #Replace all emojis
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            
        #Replace @username to USER:
        tweet = re.sub(userPattern, ' USER', tweet)
        
        # Replace all non alphabets
        tweet = re.sub(alphaPattern, " ", tweet)
        
        # Replace 3 or more consecutive letters by 2 Letter
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        
        # Removing punctuations
        all_char_list = []
        all_char_list = [char for char in tweet if char not in string.punctuation]
        tweet = ''.join(all_char_list)
        
        #removing stopwords
        tweetwords = ''
        for word in tweet.split():
            if word not in (stopwordlist):
                if len(word)>1:
                    tweetwords += (word+ ' ')
        
        processedText.append(tweetwords)
        
    return processedText
            
# function to return a plotly bar chart showing the number of positives and negatives
def plot(df):
    st.write("Positive vs Negative Sentiment %")
    positive = round(np.count_nonzero(df['Sentiment'] == "Positive")/len(df['Sentiment'])*100,2)
    negative = round(np.count_nonzero(df['Sentiment'] == "Negative")/len(df['Sentiment'])*100,2)
    values = [positive, negative]
    labels = ['Positive', 'Negative']

    # values = np.array([positive,negative])
    # myexplode = [0.2,0]
    mycolours = ["darkblue", "royalblue"]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, text = [f'{positive}%', f'{negative}%'], textposition = 'outside')])

    # fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
    #               marker=dict(colors=mycolours))

    fig.update_layout(margin=dict(t = 0, b=0, l=0, r=0), boxgap=0.25,
    boxgroupgap=0.25)
    # fig,ax = plt.subplots()
    # ax.pie(values, labels = labels, explode = myexplode, shadow = True, colors = mycolours)
    # ax.legend()
    # ax.set_title("Positive vs Negative Tweet (%)")
    
    st.plotly_chart(fig, use_container_width=True)

def progressbar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
st.set_page_config(layout="wide")
st.title('Sentiment Analysis')


# sidebar for text input with ; 
st.sidebar.subheader("Enter single/multiple texts separated by semicolon : ")
tweets = st.sidebar.text_area("Some samples are provided below for reference...", value = "I hate twitter; I do not like the movie; I don't feel so good; I am happy; Life is great")
cols = ["tweet"]


# sidebar button for text input
if (st.sidebar.button('Predict Sentiment')):
    st.write('Performing Sentiment Analysis')
    progressbar()
    st.write("Table Text Data & Sentiment %")
    vectoriser, model = load('models/vectoriser.pickle', 'models/Sentiment-LR.pickle')
    result_df = inference(vectoriser, model, tweets, cols)
    

    st.table(result_df)
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    plot(result_df)

# side bar download csv template to fill data points
import base64
st.sidebar.text("")
st.sidebar.text("")

if (st.sidebar.button('Download CSV Template')):
    data = [('1',"i'm fine. how about yourself?",	"29/02/2016 16:40",	"29/02/2016 16:47",	"Male", 41)]
    df = pd.DataFrame(data, columns=["id", "Text", "start", "end", "gender", "age"])
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;sentiment_template&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)


st.sidebar.text("")

#CSV upload
uploaded_file = st.sidebar.file_uploader('Upload CSV data with the template above:') 
# if uploaded_file is not None:
    
    # print(df1.head())


from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode()



def plotd():
    st.write("Duration-Gender-Age Analytics")
    st.write("...#not done")
    
     
    # st.plotly_chart(fig, use_container_width=True)s



#csv upload result button
if (st.sidebar.button('CSV Predict Sentiment')):
    df1=pd.read_csv(uploaded_file)
    # print(df1.head())
    st.write('Performing Sentiment Analysis')
    progressbar()
    text = list(df1['Text'])
    # load sentiment analysis model
    vectoriser, model = load('models/vectoriser.pickle', 'models/Sentiment-LR.pickle')
    result_df1 = predict_Excel(vectoriser, model, text)
    # st.table(result_df1)

    # calc duration of chat from start and end timestamps
    t1 = pd.to_datetime(df1['start'])
    t2 = pd.to_datetime(df1['end'])
    duration = pd.DataFrame(pd.to_timedelta((t2 - t1).dt.seconds / 60.0)).astype(int)

    # show table data from csv template upload with sentiment prediction per userchat chunk
    st.write("Table Text Data & Sentiment %")

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Text', 'Sentiment', "Prob", "Chat Duration (minutes)", "Gender", "Age"],
                    fill = dict(color='#C2D4FF'), font = dict(color = "black"),
                    align = ['left']),
        cells=dict(values=[result_df1['Text'], result_df1['Sentiment'], result_df1['Probability'], duration, df1['gender'], df1['age']],
                fill = dict(color='#F5F8FF'),
                align=['left', 'center'],font_size=12,height=30, font = dict(color = "black")))])

    fig.update_layout(
    height=250, margin=dict(t = 0, b=0, l=0, r=0), boxgap=0,
    boxgroupgap=0)

    st.plotly_chart(fig,use_container_width=True)
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    # plot positive vs negative barcharts totals
    plot(result_df1)
    
    st.write("Not Interested, Neutral and Interested Count")
    
    # classify not interested, neutral, interested based on probability criteria
    # Negative >= 40% --> Not interested
    # Negative or Positive <40% --> Neutral
    # Positive >= 40% --> Interested
    
    result_df1['Probability'] = result_df1['Probability'].astype(float)
    classify_neutrals = result_df1[result_df1['Probability'] < 0.4]
    count_neutrals = len(classify_neutrals)
    
    positives = result_df1[result_df1['Sentiment'] == 'Positive']
    classify_interested =  positives[positives['Probability'] >= 0.4]
    count_interested = len(classify_interested)

    negatives = result_df1[result_df1['Sentiment'] == 'Negative']
    classify_notinterested =  negatives[negatives['Probability'] >= 0.4]
    count_notinterested = len(classify_notinterested)

    labels = ['not interested', 'neutral', 'interested']
    values = [count_notinterested, count_neutrals, count_interested]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, text = [f'{count_notinterested}', f'{count_neutrals}', f'{count_interested}'], textposition = 'outside')])

    fig.update_layout(margin=dict(t = 0, b=0, l=0, r=0), boxgap=0.25,
    boxgroupgap=0.25)
 
    
    st.plotly_chart(fig, use_container_width=True)

    st.write("Gender Count")
    classify_males = df1[df1['gender'] == 'Male']
    count_males = len(classify_males)

    classify_females = df1[df1['gender'] == 'Female']
    count_females = len(classify_females)

    labels2 = ['Males', 'Females']
    values2 = [count_males, count_females]

    fig = go.Figure(data=[go.Bar(x=labels2, y=values2, text = [f'{count_males}', f'{count_females}'], textposition = 'outside')])
    

    fig.update_layout(margin=dict(t = 0, b=0, l=0, r=0), boxgap=0.25,
    boxgroupgap=0.25)

    st.plotly_chart(fig, use_container_width=True)

    # Age categories and count
    st.write("Age Group Breakdown")

    df1['age'] = df1['age'].astype(int)

    conditions = [
        (df1['age'] <= 5 ),
        (df1['age'] >5) & (df1['age'] <=10),
        (df1['age'] >10) & (df1['age'] <=15),
        (df1['age'] >15) & (df1['age'] <=20),
        (df1['age'] >20) & (df1['age'] <=25),
        (df1['age'] >25) & (df1['age'] <=30),
        (df1['age'] >30) & (df1['age'] <=35),
        (df1['age'] >35) & (df1['age'] <=40),
        (df1['age'] >40) & (df1['age'] <=45),
        (df1['age'] >45) & (df1['age'] <=50),
        (df1['age'] >50) & (df1['age'] <=55),
        (df1['age'] >55) & (df1['age'] <=60),
        (df1['age'] >60) & (df1['age'] <=65),
        (df1['age'] >65) & (df1['age'] <=70),
        (df1['age'] >70) & (df1['age'] <=75),
        (df1['age'] >75) & (df1['age'] <=80),
        (df1['age'] >80) & (df1['age'] <=85),
        (df1['age'] >85) & (df1['age'] <=90),
        (df1['age'] >90) & (df1['age'] <=95),
        (df1['age'] >95) & (df1['age'] <=100),
        (df1['age'] > 100)

    ]

    valuesx = ['0-5', '5-10', '11-15', '16-20', '21-25','26-30', '31-35','36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '91-95', '96-100', '100+']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df1['age_group'] = np.select(conditions, valuesx)

        
    new = df1.groupby('age_group',as_index = False).size()

    labels = list(new['age_group'])
    values = list(new['size'])

    fig = go.Figure(data=[go.Bar(x=labels, y=values, text = values, textposition = 'outside')])


    fig.update_layout(margin=dict(t = 0, b=0, l=0, r=0), boxgap=0.25,
    boxgroupgap=0.25)

    st.plotly_chart(fig, use_container_width=True)



    #df1.groupby(['State','Product'],as_index = False).count().pivot('State','Product').fillna(0)


        
    

    # def top5GA(dfg1_):
    #     new = dfg1_.groupby('AGE_GROUP')['STATE'].agg(MySum='sum')
    #     top5 = new.nlargest(5, 'MySum')
    #     topages = top5.index.tolist()
    #     dfg1_ = dfg1_[dfg1_['AGE_GROUP'].isin(topages)]
    #     name_sort = {'0-5': 0,'5-10':1,'11-15':2, '16-20':3, '21-25':4,'26-30':5, '31-35':6,'36-40':7, '41-45':8, '46-50':9, '51-55':10, '56-60':11, '61-65':12, '66-70':13, '71-75':14, '76-80':15, '81-85':16, '86-90':17, '91-95':18, '96-100':19, '100+':20}
    #     dfg1_['name_sort'] = dfg1_.AGE_GROUP.map(name_sort)
    #     dfg1_ = dfg1_.sort_values(['name_sort', 'GENDER'])

    #     return dfg1_




