"""
Plan: input a channel's id
Then collect all the video ids
if we have the assigned transcripts,then we should not have any audio files
If we have no assigned transcripts, but
    we have normal transcripts and audio files, then make the assigned transcripts
    we have audio files, we should make the normal and assigned transcripts
    we have nothing, then we should download the audio files and proceed from there
"""


import yt_dlp
import whisperx_transcription
import warnings
import json
import os
import requests
import time
import yaml
import random
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
import librosa
import torch
from pathlib import Path
from typing import Dict, List, Any
from peterson_sphere_function import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def make_whisper_transcript(audiofile_address, min_speakers, max_speakers, HF_TOKEN, whisperx_transcript_address, video_id):
    try:
        result = whisperx_transcription.transcribe_basic(
            audio_file=audiofile_address,
            model_size="tiny",  # or "medium", "small", etc.
            device="cpu",  # or "cpu" if no GPU, original "cuda"
            compute_type="int8", # This exists because I was running on my small device,
            min_speakers = min_speakers,
            max_speakers = max_speakers,
            hf_token = HF_TOKEN
        )

        print('I am now this far. This should ')

        if result["success"]:
            print(f"Transcription completed!")
            print(f"Language detected: {result['language']}")
            print("\nTranscript with speakers:")
            print("-" * 50)

            for segment in result["segments"]:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment["text"]
                speaker = segment.get("speaker", "Unknown")

            result = save_whisperx_transcript_to_yaml(result, whisperx_transcript_address, video_id)
        else:
            print("Transcription failed!")

    except Exception as e:
        print(f"Error: {e}")

    return result


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
torchaudio.set_audio_backend("ffmpeg")  # or try "sox"
# Clean up the console output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TF info/warning messages
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", category=FutureWarning)  # This will catch it


keys = open('keys.txt')
key_list = keys.readlines()
API_KEY = key_list[0]
HF_TOKEN = key_list[1].strip()
CLAUDE_TOKEN = key_list[2]

print('WARNING \n This script will transcribe many long audio files using whisperx and it may take a long time to execute fully')
print('Input the desired youtube channel ID:                (for example, UCL_f53ZEJxp8TtlOkHwMV9Q)')
channel_id = input()

if channel_id == 'PVK':
    channel_id = 'UCGsDIP_K6J6VSTqlq-9IPlg'
else:
    pass

print(f'Now starting to make all transcripts from the channel: "{channel_id}"')

### Parameters:
max_results = 5
database_name = r'database'
voice_sample_directory = 'voice_samples'


## Get info about the channel using the channel id, to get the uploads playlist id
channel_info = get_channel_info(api_key=API_KEY, channel_id=channel_id)
playlist_id = channel_info['uploadsPlaylistId']
channel_name = channel_info['title']
print(f'Channel name: {channel_name}')


## Get the full list of uploads by that channel, each video has a dictionary describing it

existing_saved_files = get_list_of_saved_local_transcripts(database_name, channel_name)


## save the list to a file
print("If you want to download a new set of video information (THIS MIGHT TAKE MULTIPLE HOURS) then enter a lowercase Y: 'y', otherwise enter anything else.")
do_we_get_new_list = input()
if do_we_get_new_list == 'y':
    all_videos_list = get_all_channel_videos(API_KEY, playlist_id, max_videos=None)
    dumpfile_name = 'saved_videolists/'+channel_id+'.txt'
    with open(dumpfile_name, 'w') as f:
        f.write(json.dumps(all_videos_list))
else:
    try:
        print('Trying to get the video list from the saved dump file')
        dumpfile_name = 'saved_videolists/' + channel_id + '.txt'
        with open(dumpfile_name, 'r') as f:
            all_videos_list = json.loads(f.read())
    except Exception as e:
        print(f"Error: {e}")

print('')
print(f"There are {len(all_videos_list)} videos in this channel to be analysed")


def process_file_from_name(video, parameters):
    script_starting_time = datetime.now()
    ## Get the videoId for YouTube and filename for internal reference
    video_id = video['videoId'] #e.g. 7OAOksRVmpU
    channel_name, API_KEY, HF_TOKEN, CLAUDE_TOKEN, database_name, voice_sample_directory = parameters
    filename = make_video_filename(video) #e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    ## Check if the video already has its transcript locally saved
    if 'assigned' in existing_saved_files.get(video_id, []):
        print('')
        try:
            print(f"{filename['filename']} automatic transcript already locally saved")
            pass
        except Exception as e:
            print(f"Error: {e}")

    elif 'audio' in existing_saved_files.get(video_id, []):
        print('')
        try:
            filenamex = filename['filename']
            print(f"{filenamex} audiofile is saved, now making the transcript using whisperx:")

            ## Get datapoints
            filename = make_video_filename(video)  # e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            folder_path = guarantee_directories(database_name, channel_name,filename)  # e.g. C:\Users\Gusta\Desktop\Peterson_Sphere_Local\PaulVanderKlay\2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            audio_filename = 'audio_' + filename['filename']  # e.g. audio_2025_05_12_7OAOksRVmpU_MartinShawContinuedU

            ## Make the transcripts and save them
            folder_name = get_foldername_from_video_id(database_name, channel_name, video_id)
            folder_address = database_name + '/' + channel_name + '/' + folder_name
            full_filelist = os.listdir(folder_address)
            audiofilename = [x for x in full_filelist if x.split('_')[0] == 'audio'][0]
            audiofile_address = folder_address + '/' + audiofilename

            print(f"Starting the transcription, diarisation process now at {datetime.now()}")
            whisperx_transcript_address = folder_address + '/whisperx_' + folder_name + '.yaml'
            min_speakers = 1
            max_speakers = 6
            transcript = make_whisper_transcript(audiofile_address, min_speakers, max_speakers, HF_TOKEN,whisperx_transcript_address, video_id)
            shortened_transcript = shorten_transcript(transcript)
            assigned_through_samples = assign_speakers_through_audio_comparison(audiofile_address, shortened_transcript,voice_sample_directory)
            assigned_through_API = assign_speakers_through_api_judgement(assigned_through_samples,voice_sample_directory,assigned_through_samples['metadata']['audio_file'], CLAUDE_TOKEN, API_KEY)

            assigned_transcript_address = folder_address + '/assigned_' + filename['filename'] + '.yaml'
            save_whisperx_transcript_to_yaml(assigned_through_API, assigned_transcript_address, video_id)

            ## Remove the audio file, which is typically quite large
            os.remove(audiofile_address)

        except Exception as e:
            print(f"Error: {e}")


    else:
        print('')
        try:
            filenamex = filename['filename']
            print(f'{filenamex} has no local data saved, now downloading audio and making transcript:')

            ## Download the audio
            filename = make_video_filename(video)  # e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            folder_path = guarantee_directories(database_name, channel_name,filename)  # e.g. C:\Users\Gusta\Desktop\Peterson_Sphere_Local\PaulVanderKlay\2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            audio_filename = 'audio_' + filename['filename']  # e.g. audio_2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            extract_audio(video_url, folder_path, audio_filename)
            print(f"Saved as audio file: {video['publishedAt'], video['title']}")

            ## Make the transcripts and save them
            folder_address = database_name + '/' + channel_name + '/' + folder_name
            print(folder_address)
            full_filelist = os.listdir(folder_address)
            print(full_filelist)
            audiofilename = [x for x in full_filelist if x.split('_')[0] == 'audio'][0]
            print(audiofilename)
            audiofile_address = folder_address + '/' + audiofilename
            print(audiofile_address)

            print(f"Starting the transcription, diarisation process now at {datetime.now()}")

            whisperx_transcript_address = folder_address + '/whisperx_' + folder_name + '.yaml'
            min_speakers = 1
            max_speakers = 6
            transcript = make_whisper_transcript(audiofile_address, min_speakers, max_speakers, HF_TOKEN, whisperx_transcript_address, video_id)
            print(f"Done with transcription now at {time.now()}")
            shortened_transcript = shorten_transcript(transcript)
            assigned_through_samples = assign_speakers_through_audio_comparison(audiofile_address, shortened_transcript, voice_sample_directory)
            assigned_through_API = assign_speakers_through_api_judgement(assigned_through_samples,voice_sample_directory,assigned_through_samples['metadata']['audio_file'],CLAUDE_TOKEN,API_KEY)

            assigned_transcript_address = folder_address + '/assigned_' + filename['filename'] + '.yaml'
            save_whisperx_transcript_to_yaml(assigned_through_API, assigned_transcript_address, video_id)

            ## Remove the audio file, which is typically quite large
            os.remove(audiofile_address)

        except Exception as e:
            print(f"Error: {e}")
    script_ending_time = datetime.now()
    script_running_time = script_ending_time-script_starting_time
    return filename, script_running_time


## Iterate through the list, collecting, formatting and saving the transcripts
##for video in all_videos_list:
##    process_file_from_name(video, parameters)

def transcribe_files_parallel(all_videos_list, parameters, max_threads=8):
    # Load model once for reuse, disables GPU here for CPU usage

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_file_from_name, video, parameters) for video in all_videos_list]
        for future in as_completed(futures):
            filename, script_running_time = future.result()
            print(f"Completed: {filename} in {script_running_time}")


parameters = (channel_name, API_KEY, HF_TOKEN, CLAUDE_TOKEN, database_name, voice_sample_directory)
transcribe_files_parallel(all_videos_list, parameters, max_threads=8)
print(f'This channel, {channel_name}, now has all data saved locally! Congratulations!')