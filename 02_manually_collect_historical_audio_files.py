from peterson_sphere_function import *
import yt_dlp

"""
This script is meant to be called through the terminal
with manual input of the channel id
to create directories for and collect transcripts from
ALL videos from that channel
"""


with open('keys.txt', 'r') as file:
    API_KEY = str(file.read().strip())

print('WARNING \n This script will collect a vast amount of data using the youtube_transcript_api and it may take a long time to execute fully')
print('Input the desired youtube channel ID:                (for example, UCL_f53ZEJxp8TtlOkHwMV9Q)')
channel_id = input()

print(f'Now starting to collect all transcripts from the YouTube channel: "{channel_id}"')


#channel_id = 'UCL_f53ZEJxp8TtlOkHwMV9Q'


### Parameters:
max_results = 5
database_name = r'database'

## Get info about the channel using the channel id, to get the uploads playlist id
channel_info = get_channel_info(api_key=API_KEY, channel_id=channel_id)
print(channel_info)
playlist_id = channel_info['uploadsPlaylistId']
channel_name = channel_info['title']

existing_saved_files = get_list_of_saved_local_transcripts(database_name, channel_name)

## Get the full list of uploads by that channel, each video has a dictionary describing it
all_videos_list = get_all_channel_videos(API_KEY, playlist_id, max_videos=None)


## Iterate through the list, collecting, formatting and saving the transcripts
for video in all_videos_list:

    print(video)
    ## Get the videoId for YouTube and filename for internal reference
    video_id = video['videoId'] #e.g. 7OAOksRVmpU
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    filename = make_video_filename(video) #e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU

    ## Check if the video already has its transcript locally saved
    if 'audio' in existing_saved_files.get(video_id, []):
        print(f"{filename['filename']} audio already locally saved")
        pass

    else:
        ## Get/make a bunch of addresses for directories
        folder_path = guarantee_directories(database_name, channel_name, filename)  # e.g. C:\Users\Gusta\Desktop\Peterson_Sphere_Local\PaulVanderKlay\2025_05_12_7OAOksRVmpU_MartinShawContinuedU
        audio_filename = 'audio_' + filename['filename']  # e.g. audio_2025_05_12_7OAOksRVmpU_MartinShawContinuedU

        extract_audio(video_url, folder_path, audio_filename)

        print(f"Saved as audio file: {video['publishedAt'], video['title']}")

print(f'All audio collected for the channel: "{channel_id}"')