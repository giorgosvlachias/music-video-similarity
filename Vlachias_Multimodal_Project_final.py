import pafy
import requests
from apiclient.discovery import build
import re
import subprocess
import os
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import matplotlib.pyplot as plt
import sklearn
import librosa.display
import video_features as vf
import plotly
import plotly.graph_objs as go


# youtube metadata as json of pafy objects
playlist_url = "https://www.youtube.com/playlist?list=PLUuOv5NaKEgTUbA4nzJpiID5xGOWyY0f_"
#playlist_url = "https://www.youtube.com/watch?v=a_426RiwST8&list=PLUuOv5NaKEgSuWS8l7u9m2F8eQnOboiqy"
playlist = pafy.get_playlist(playlist_url)

print(playlist['title'])
print(playlist['author'])
print(len(playlist['items']))

print(playlist['items'][0]['pafy'])
print(playlist['items'][0]['playlist_meta']['title'])
print(playlist['items'][0]['playlist_meta']['encrypted_id'])
#print(playlist['items'][0]['pafy'].title
#playlist['items'][0]['pafy'].videoid

# storing music videos, audio, videos
if not os.path.exists("music_videos"):
	os.mkdir("music_videos")
	print("Music Video Directory created")
else:
	print("Music Video Directory already exists")
	
if not os.path.exists("audio"):
	os.mkdir("audio")
	print("Audio Directory created")
else:
	print("Audio Directory already exists")

if not os.path.exists("videos"):
	os.mkdir("videos")
	print("Video Directory created")
else:
	print("Video Directory already exists")

for i in range(0,len(playlist['items'])):
	name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
	filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
	
	playlist['items'][i]['pafy'].getbest(preftype='mp4').download("music_videos/"+filename+".mp4")
	command_audio = "ffmpeg -i music_videos/"+filename+".mp4 -ab 160k -ac 2 -ar 44100 -vn audio/"+filename+".wav"
	subprocess.call(command_audio, shell = True, executable = "/bin/bash")
	command_video = "ffmpeg -i music_videos/"+filename+".mp4 -codec copy -an videos/"+filename+".mkv"
	subprocess.call(command_video, shell = True, executable = "/bin/bash")

# spotify API authorization
access_token='BQCh09HV2-CcVPcO2xogkw-_NjqX-c7T2cxEDx8qDhTTEVhHYmhI4dd0nVcgpu6UwiGa-NmPiZGTUJsRebrzYdejRN44z9C5J9JukNxucSkdcxy-RbWqaak57Rd_cHIDlYAEmyuacAlezBIcs7ZMRHYBtsx0g-UJEXJjx1xkSPVEStE_fb3y&refresh_token=AQAXgluiQK2znTIZgdQExCrCLh-TCv34aWFm_sUJc-DOQ8IeygydPX-Zo3zUQNddzkuPsqlV-2JFbgGs2kXCY02cCwt6E47mbJaBvoaStRgjchEWYWsF_4HKd2Soe6lPV43qgA'

h = {'Authorization': 'Bearer '+access_token}

#searching for spotify track ids
ids = "ids="
for i in range(0,len(playlist['items'])):
    title = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title']).split(' - ')
    if len(title) == 2:
        title[0].replace(' ','%20')
        track = title[1].replace(' ','%20')
        artist = title[0].replace(' ','%20')
        spotify_search_url = "https://api.spotify.com/v1/search?q=track:"+track+"%20artist:"+artist+"%20&type=track"
    else:
        track = title[0].replace(' ','%20')
        spotify_search_url = "https://api.spotify.com/v1/search?q=track:"+track+"%20&type=track"
            
    search = requests.get(spotify_search_url, headers = h)
    data = search.json()
    if i == 0:
        print(type(data))
        print(data)
    if data['tracks']['items']==[]:
        ids = ids+""
    else:
        spotify_artist_id = data['tracks']['items'][0]['artists'][0]['id']
        spotify_artist_name = data['tracks']['items'][0]['artists'][0]['name']
        spotify_track_uri = data['tracks']['items'][0]['uri'].split(':')
        spotify_track_id = spotify_track_uri[2]
        ids = ids+spotify_track_id+","

# requesting audio features for track ids
spotify_audio_features_url = "https://api.spotify.com/v1/audio-features/?"+ids
h = {'Authorization': 'Bearer '+access_token}
response = requests.get(spotify_audio_features_url, headers = h)
print(response.status_code)

data = response.json()
print(type(data))
print(data['audio_features'][0])

#Audio Feature Extraction
# 13 mfccs + 13 deltas + 13 deltadeltas) x 2 statistical measures mean, var
feats = np.zeros((len(playlist['items']), 78))

for i in range(0,len(playlist['items'])):
    name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
    filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
    [s, Fs] = librosa.load("audio/"+filename+".wav")
    
    #sr, data = wavfile.read("audio/"+filename+".wav")
    #data = np.double(data)
    #data = data/ (2.0 **15)
    #data = (data - data.mean())/((np.abs(data)).max()+ 0.0000000001)
    
    mfccs = librosa.feature.mfcc(y=s, sr=Fs, n_mfcc=13)
    deltas = librosa.feature.delta(mfccs)
    deltadeltas = librosa.feature.delta(deltas,order=2)
    feats[i,0:13] = np.mean(mfccs, axis=1)
    feats[i,13:26] = np.var(mfccs, axis=1)
    feats[i,26:39] = np.mean(deltas, axis=1)
    feats[i,39:52] = np.var(deltas, axis=1)
    feats[i,52:65] = np.mean(deltadeltas, axis=1)
    feats[i,65:78] = np.var(deltadeltas, axis=1)
    
    if i == 0:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=Fs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        
        mfccs_scaled = sklearn.preprocessing.scale(mfccs, axis=1)
        print(mfccs_scaled.mean(axis=1))
        print(mfccs_scaled.var(axis=1))
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs_scaled, sr=Fs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC scaled')
        plt.tight_layout()
        
        mfccs_htk = librosa.feature.mfcc(y=s, sr=Fs, n_mfcc=13, dct_type=3)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs_htk, sr=Fs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC DCT type-3')
        plt.tight_layout()
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(deltas, sr=Fs, x_axis='time')
        plt.colorbar()
        plt.title('Deltas of MFCCs across frames')
        plt.tight_layout()
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(deltadeltas, sr=Fs, x_axis='time')
        plt.colorbar()
        plt.title('2nd order Deltas of MFCCs across frames')
        plt.tight_layout()

chroma = np.zeros((len(playlist['items']), 24))

for i in range(0,len(playlist['items'])):
    name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
    filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
    [s, Fs] = librosa.load("audio/"+filename+".wav")
    
    #sr, data = wavfile.read("audio/"+filename+".wav")
    #data = np.double(data)
    #data = data/ (2.0 **15)
    #data = (data - data.mean())/((np.abs(data)).max()+ 0.0000000001)
    
    chromagram = librosa.feature.chroma_stft(y=s, sr=Fs)
    chroma[i,0:12] = np.mean(chromagram, axis=1)

    chromagram_cens = librosa.feature.chroma_cens(y=s, sr=Fs)
    chroma[i,12:24] = np.mean(chromagram_cens, axis=1)
    
    if i == 0:
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma',cmap='coolwarm')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        
        plt.figure(figsize=(15,5))
        librosa.display.specshow(chromagram_cens, x_axis='time',y_axis='chroma',cmap='coolwarm')
        plt.colorbar()
        plt.title('Chromagram CENS')
        plt.tight_layout()
        
        plt.figure(figsize=(15,5))
        librosa.display.specshow(chromagram[:,1500:1750], x_axis='time',y_axis='chroma',cmap='coolwarm')
        plt.colorbar()
        plt.title('Chromagram, ~5sec')
        plt.tight_layout()
        
        plt.figure(figsize=(15,5))
        librosa.display.specshow(chromagram_cens[:,1500:1750], x_axis='time',y_axis='chroma',cmap='coolwarm')
        plt.colorbar()
        plt.title('Chromagram CENS, ~5sec')
        plt.tight_layout()

# beat locations and global tempo estimation

for i in range(0,len(playlist['items'])):
    name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
    filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
    [s, Fs] = librosa.load("audio/"+filename+".wav")
    
    #sr, data = wavfile.read("audio/"+filename+".wav")
    #data = np.double(data)
    #data = data/ (2.0 **15)
    #data = (data - data.mean())/((np.abs(data)).max()+ 0.0000000001)
    
    tempo, beat_times = librosa.beat.beat_track(y=s, sr=Fs, units='time')
    print(tempo)
    #print(beat_times)
    
    if i == 0:
        plt.figure(figsize=(15, 5))
        librosa.display.waveplot(s[0:120000])
        plt.vlines(beat_times[0:16], -1, 1, color='r')
        plt.ylim(-1,1)
        plt.title('Beat Tracking')
        plt.tight_layout()


# Video Feature Extraction
for i in range(0,1):
    name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
    filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
    v = vf.VideoFeatureExtractor(["hogs"], resize_width=300, step = 0.25)
    f, t, fn = v.extract_features("music_videos/"+filename+".mp4")

print(f.shape)
print(t.shape)
print(len(fn))

for i in range(0,len(playlist['items'])):
    name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
    filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
    v = vf.VideoFeatureExtractor(["colors"], resize_width=300, step = 0.25)
    f, t, fn = v.extract_features("music_videos/"+filename+".mp4")

print(f.shape)
print(t.shape)
print(len(fn))

fig = go.Figure()
p = go.Scatter(x=fn, y=f[10,:], name=filename)
fig = go.Figure(data = [p])
plotly.offline.plot(fig, filename="temp.html", auto_open=True)

print(f.shape)
print(t.shape)
print(len(fn))
