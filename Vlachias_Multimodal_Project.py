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
import librosa.display
import video_features as vf

# youtube metadata as json of pafy objects
playlist_url = "https://www.youtube.com/playlist?list=PLUuOv5NaKEgTUbA4nzJpiID5xGOWyY0f_"
#playlist_url = "https://www.youtube.com/watch?v=a_426RiwST8&list=PLUuOv5NaKEgSuWS8l7u9m2F8eQnOboiqy"
playlist = pafy.get_playlist(playlist_url)

print(playlist['title'])
print(playlist['author'])
print(len(playlist['items']))

print(playlist['items'][74]['pafy'])
print(playlist['items'][74]['playlist_meta']['title'])
print(playlist['items'][74]['playlist_meta']['encrypted_id'])
#print(playlist['items'][74]['pafy'].title
#playlist['items'][74]['pafy'].videoid

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
	title[0].replace(' ','%20')
	track = title[1].replace(' ','%20')
	artist = title[0].replace(' ','%20')
	spotify_search_url = "https://api.spotify.com/v1/search?q=track:"+track+"%20artist:"+artist+"%20&type=track"
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
	mfccs = np.transpose(librosa.feature.mfcc(y=s,sr=Fs,n_mfcc=13))
	deltas = librosa.feature.delta(mfccs)
	deltadeltas = librosa.feature.delta(deltas)
	feats[i,0:12] = np.mean(mfccs)
	feats[i,13:25] = np.var(mfccs)
	feats[i,26:38] = np.mean(deltas)
	feats[i,39:51] = np.var(deltas)
	feats[i,52:64] = np.mean(deltadeltas)
	feats[i,65:77] = np.var(deltadeltas)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

# Video Feature Extraction
for i in range(0,len(playlist['items'])):
	name = re.sub(' [\{\(\[].*?[\)\}\]]', '', playlist['items'][i]['playlist_meta']['title'])
	filename = re.sub("[,.!?'|&]", '', name).replace(' ', '_')
	v = vf.VideoFeatureExtractor(["colors", "lbps", "hog","flow"], resize_width=300, step = 0.25)
	f, t, fn = v.extract_features("videos/"+filename+".mkv")

print(f.shape)
print(t.shape)
print(len(fn))
