import streamlit as st
import librosa
import numpy as np
import pickle
import hashlib
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import maximum_filter


# PAGE SETTINGS

st.set_page_config(
    page_title="Mini Shazam",
    layout="wide"
)


# LOAD DATABASE

@st.cache_resource
def load_database():

    with open("database.pkl","rb") as f:
        database = pickle.load(f)

    return database


database = load_database()

# AUDIO PROCESSING

def load_audio(file):

    y,sr = librosa.load(
        file,
        sr=22050,
        mono=True
    )

    return y,sr



def compute_spectrogram(y):

    S = librosa.stft(
        y,
        n_fft=2048,
        hop_length=512
    )

    S_db = librosa.amplitude_to_db(
        np.abs(S),
        ref=np.max
    )

    return S_db



def find_peaks(
    S_db,
    neighborhood_size=30,
    amp_min=-20
):

    local_max = (
        maximum_filter(
            S_db,
            size=neighborhood_size
        )
        ==
        S_db
    )


    detected = np.logical_and(
        local_max,
        S_db > amp_min
    )


    freq,time = np.where(detected)


    peaks=list(
        zip(time,freq)
    )


    return peaks




# HASH GENERATION


def generate_hashes(
        peaks,
        fan_value=10):

    hashes=[]


    for i in range(len(peaks)):


        t1,f1=peaks[i]


        for j in range(1,fan_value+1):

            if i+j < len(peaks):

                t2,f2=peaks[i+j]


                dt=t2-t1


                if dt>0:

                    key=f"{f1}|{f2}|{dt}"


                    h=hashlib.sha1(
                        key.encode()
                    ).hexdigest()


                    hashes.append(
                        (h,t1)
                    )

    return hashes



# MATCHING


def match_song(hashes):


    votes={}
    offsets=[]


    for h,tq in hashes:


        if h in database:


            for song,td in database[h]:


                offset=td-tq

                offsets.append(offset)


                if song not in votes:
                    votes[song]={}


                if offset not in votes[song]:
                    votes[song][offset]=0


                votes[song][offset]+=1



    if len(votes)==0:
        return "No match", len(hashes), []



    score={}


    for song,data in votes.items():

        score[song]=max(
            data.values()
        )



    winner = max(
    score,
    key=score.get
    )
    
    return (
        winner,
        score[winner],
        offsets
    )

# COMPLETE RECOGNITION


def recognize(file):


    y,sr=load_audio(file)


    S=compute_spectrogram(y)


    peaks=find_peaks(S)


    hashes=generate_hashes(peaks)


    prediction,votes,offsets=match_song(
        hashes
    )


    return (
        prediction,
        votes,
        offsets,
        S,
        peaks
    )





# VISUALS


def show_spectrogram(S):

    fig,ax=plt.subplots()

    ax.imshow(
        S,
        origin="lower",
        aspect="auto"
    )

    ax.set_title(
        "Spectrogram"
    )

    return fig



def show_constellation(peaks):

    fig,ax=plt.subplots()


    t=[p[0] for p in peaks]
    f=[p[1] for p in peaks]


    ax.scatter(
        t,
        f,
        s=5
    )


    ax.set_title(
        "Constellation Map"
    )

    return fig



def show_hist(offsets):

    fig,ax=plt.subplots()


    ax.hist(
        offsets,
        bins=50
    )

    ax.set_title(
        "Offset Histogram"
    )


    return fig



# UI


st.title(
    "🎵 EE200 Mini Shazam"
)


tab1,tab2,tab3 = st.tabs(
[
"Library",
"Identify Song",
"Batch Mode"
]
)



# LIBRARY

with tab1:


    st.header(
        "Song Database"
    )


    songs=set()


    for value in database.values():

        for item in value:

            songs.add(
                os.path.splitext(item[0])[0]
            )


    st.metric(
        "Songs Indexed",
        len(songs)
    )


    st.success(
        "Database Loaded Successfully"
    )



# SINGLE CLIP

with tab2:


    st.header(
        "Identify Audio"
    )


    file=st.file_uploader(
        "Upload audio",
        type=[
            "mp3",
            "wav",
            "m4a"
        ]
    )


    if file:


        with open(
            "temp.wav",
            "wb"
        ) as f:

            f.write(
                file.read()
            )


        prediction,votes,offsets,S,peaks = recognize(
            "temp.wav"
        )


        c1,c2,c3=st.columns(3)


        c1.metric(
            "Prediction",
            prediction
        )


        c2.metric(
            "Votes",
            votes
        )


        c3.metric(
            "Peaks",
            len(peaks)
        )


        st.pyplot(
            show_spectrogram(S)
        )


        st.pyplot(
            show_constellation(peaks)
        )


        st.pyplot(
            show_hist(offsets)
        )




# BATCH

with tab3:


    st.header(
        "Batch Recognition"
    )


    files=st.file_uploader(
        "Upload multiple clips",
        accept_multiple_files=True,
        type=[
            "mp3",
            "wav"
        ]
    )


    if files:


        result=[]


        for f in files:


            with open(
                "temp.wav",
                "wb"
            ) as x:

                x.write(
                    f.read()
                )


            pred,_,_,_,_=recognize(
                "temp.wav"
            )


            result.append(
                [
                    f.name,
                    pred
                ]
            )



        df=pd.DataFrame(
            result,
            columns=[
                "filename",
                "prediction"
            ]
        )


        st.dataframe(df)



        st.download_button(
            "Download results.csv",
            df.to_csv(index=False),
            "results.csv"
        )
