"""
Plan: input a channel's id
Then collect all the video ids
if we have the assigned transcripts,then we should not have any audio files
If we have no assigned transcripts, but
    we have normal transcripts and audio files, then make the assigned transcripts
    we have audio files, we should make the normal and assigned transcripts
    we have nothing, then we should download the audio files and proceed from there
"""

from peterson_sphere_function import *
import yt_dlp
from whisperx_transcription import transcribe_basic
import warnings

keys = open('keys.txt')
key_list = keys.readlines()
API_KEY = key_list[0]
HF_TOKEN = key_list[1].strip()
CLAUDE_TOKEN = key_list[2]

print('WARNING \n This script will transcribe many long audio files using whisperx and it may take a long time to execute fully')
print('Input the desired youtube channel ID:                (for example, UCL_f53ZEJxp8TtlOkHwMV9Q)')
channel_id = input()

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
all_videos_list = get_all_channel_videos(API_KEY, playlist_id, max_videos=None)


## Iterate through the list, collecting, formatting and saving the transcripts
for video in all_videos_list:

    ## Get the videoId for YouTube and filename for internal reference
    video_id = video['videoId'] #e.g. 7OAOksRVmpU
    filename = make_video_filename(video) #e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    ## Check if the video already has its transcript locally saved
    if 'assigned' in existing_saved_files.get(video_id, []):
        try:
            print(f"{filename['filename']} automatic transcript already locally saved")
            pass
        except Exception as e:
            print(f"Error: {e}")

    elif 'audio' in existing_saved_files.get(video_id, []):
        try:
            print(f'{filename['filename']} audiofile is saved, now making the transcript using whisperx:')

            ## Download the audio
            filename = make_video_filename(video)  # e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            folder_path = guarantee_directories(database_name, channel_name,filename)  # e.g. C:\Users\Gusta\Desktop\Peterson_Sphere_Local\PaulVanderKlay\2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            audio_filename = 'audio_' + filename['filename']  # e.g. audio_2025_05_12_7OAOksRVmpU_MartinShawContinuedU

            ## Make the transcripts and save them
            folder_address = database_name + '/' + channel_name + '/' + folder_name
            full_filelist = os.listdir(folder_address)
            audiofilename = [x for x in full_filelist if x.split('_')[0] == 'audio'][0]
            audiofile_address = folder_address + '/' + audiofilename

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
        try:
            print(f'{filename['filename']} has no local data saved, now downloading audio and making transcript:')

            ## Download the audio
            filename = make_video_filename(video)  # e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            folder_path = guarantee_directories(database_name, channel_name,filename)  # e.g. C:\Users\Gusta\Desktop\Peterson_Sphere_Local\PaulVanderKlay\2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            audio_filename = 'audio_' + filename['filename']  # e.g. audio_2025_05_12_7OAOksRVmpU_MartinShawContinuedU
            extract_audio(video_url, folder_path, audio_filename)
            print(f"Saved as audio file: {video['publishedAt'], video['title']}")

            ## Make the transcripts and save them
            folder_address = database_name + '/' + channel_name + '/' + folder_name
            full_filelist = os.listdir(folder_address)
            audiofilename = [x for x in full_filelist if x.split('_')[0] == 'audio'][0]
            audiofile_address = folder_address + '/' + audiofilename

            whisperx_transcript_address = folder_address + '/whisperx_' + folder_name + '.yaml'
            min_speakers = 1
            max_speakers = 6
            transcript = make_whisper_transcript(audiofile_address, min_speakers, max_speakers, HF_TOKEN, whisperx_transcript_address, video_id)
            shortened_transcript = shorten_transcript(transcript)
            assigned_through_samples = assign_speakers_through_audio_comparison(audiofile_address, shortened_transcript, voice_sample_directory)
            assigned_through_API = assign_speakers_through_api_judgement(assigned_through_samples,voice_sample_directory,assigned_through_samples['metadata']['audio_file'],CLAUDE_TOKEN,API_KEY)

            assigned_transcript_address = folder_address + '/assigned_' + filename['filename'] + '.yaml'
            save_whisperx_transcript_to_yaml(assigned_through_API, assigned_transcript_address, video_id)

            ## Remove the audio file, which is typically quite large
            os.remove(audiofile_address)

        except Exception as e:
            print(f"Error: {e}")

print(f'This channel, {channel_name}, now has all data saved locally! Congratulations!')