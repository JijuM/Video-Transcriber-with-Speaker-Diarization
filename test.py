num_speakers = 2

language = 'English'

model_size = 'large'


model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

path="test.mp4"

import subprocess

command = f"ffmpeg -i {path} -ab 160k -ac 2 -ar 44100 -vn audio.wav"

subprocess.call(command, shell=True)
if path[-3:] != 'wav':
  subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
  path = 'audio.wav'

import whisperx
import os
import gc
import torch
torch.cuda.is_available()

device="cuda"
batch_size=4
compute_type="float16"
import torch
torch.cuda.is_available()
audio_file=path

model = whisperx.load_model("large-v2",device,compute_type=compute_type)

audio=whisperx.load_audio(audio_file)
result=model.transcribe(audio,batch_size)

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"],
                                              device=device)

result = whisperx.align(result["segments"], model_a,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False)

lis=[]
sentences=[]
for i in range(0,len(result['segments'])):
    lis.append(result["segments"][i].get('words'))
    sentences.append(result["segments"][i].get('text'))
    
    sentences.append(result["segments"][i].get('start'))
    sentences.append(result["segments"][i].get('end'))


    

# for i in range(0,len(lis)):
#   for j in range(0,len(lis[i])):
#   #   print(lis[i][j].get('word'))
#   #   print(lis[i][j].get('start'))
#   #   print(lis[i][j].get('end'))
#   #   print("---------------------")
#   # print("\n")
segments=result["segments"]


for i in sentences:
  print(i)

with contextlib.closing(wave.open('audio.wav','r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

  audio = Audio()


def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)

    # Convert waveform to single channel
    waveform = waveform.mean(dim=0, keepdim=True)

    return embedding_model(waveform.unsqueeze(0))

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)


embeddings = np.nan_to_num(embeddings)

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
  return datetime.timedelta(seconds=round(secs))

f = open("transcript.txt", "w",encoding="UTF-8")

for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
  f.write(segment["text"][0:] + ' ')
f.close()

print(open('transcript.txt','r',encoding="UTF-8").read())



def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


# # import cv2
# # from ffpyplayer.player import MediaPlayer
# # file = audio_file
# # cap = cv2.VideoCapture(file)
# # index=int(seconds)*1000
# # # cap.set(cv2.CAP_PROP_POS_FRAMES, index)
# # cap.set(cv2.CAP_PROP_POS_MSEC, index)

# # player = MediaPlayer(file,ff_opts={ 'ss': int(seconds)})


# # if not cap.isOpened():
# #     print("Error opening video file")

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     audio_frame, val = player.get_frame()
# #     print(index)
# #     print(audio_frame)
# #     if ret:
# #         if cv2.waitKey(43) & 0xFF == ord('q'):
# #             break
# #         cv2.imshow('Frame', frame)
# #         if val != 'eof' and audio_frame is not None:
# #             img, t = audio_frame
# #     else:
# #         break

# # cap.release()

# # # Close all the frames
# # cv2.destroyAllWindows()

# # print("Do you want word wise segments as well? Y/N")
# # response=input()

import vlc
import os
from time import sleep

os.add_dll_directory(r'C:/Program Files/VideoLAN/VLC')  
Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new("test.mp4")

# Set the start and end times of the video
print("enter the timestamp")
secs=input()
endsecs=input()

if ":" in str(secs):
  secs=get_sec(secs)
  print(secs)
  endsecs=get_sec(endsecs)
else:
  pass

Media.add_option(f'start-time={secs}')
Media.add_option(f'stop-time={endsecs}')
Media.get_mrl()
player.set_media(Media)
player.play()

# Wait for the video to finish playing
sleep(10)
while player.is_playing():
    sleep(1)

