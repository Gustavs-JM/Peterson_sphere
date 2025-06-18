from peterson_sphere_function import *

"""
This script is meant to be called through the terminal
with manual input of the channel id
to create directories for and collect transcripts from
ALL videos from that channel
"""

print('WARNING \n This script will collect a vast amount of data using the youtube_transcript_api and it may take a long time to execute fully')
print('Input the desired youtube channel name:')
channel_id = input()

print(f'Now starting to collect all transcripts from the YouTube channel: "{channel_id}"')


#channel_id = 'UCL_f53ZEJxp8TtlOkHwMV9Q'


### Parameters:
max_results = 5
database_name = r'database'
API_KEY = "AIzaSyDqhv8tJFEVqA_vhgd1uDTo2FJEtZtKDZA"


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

    ## Get the videoId for YouTube and filename for internal reference
    video_id = video['videoId'] #e.g. 7OAOksRVmpU
    filename = make_video_filename(video) #e.g. 2025_05_12_7OAOksRVmpU_MartinShawContinuedU

    ## Check if the video already has its transcript locally saved
    if 'automaticTranscript' in existing_saved_files.get(video_id, []):
        print(f"{filename['filename']} automatic transcript already locally saved")
        pass

    else:
        ## Get/make a bunch of addresses for directories
        folder_path = guarantee_directories(database_name, channel_name, filename)  # e.g. C:\Users\Gusta\Desktop\Peterson_Sphere_Local\PaulVanderKlay\2025_05_12_7OAOksRVmpU_MartinShawContinuedU
        transcript_filename = 'automaticTranscript_' + filename['filename']  # e.g. automatic_transcript_2025_05_12_7OAOksRVmpU_MartinShawContinuedU

        ## Get the transcript from the API
        auto_yaml = save_transcript_to_yaml(video_id, video_info=video)

        ## Save the transcript to the linked directory
        save_yaml_to_address(video_filename=folder_path, filename=transcript_filename, yaml_file=auto_yaml)

        print(f"Saved as YAML automatic transcript: {video['publishedAt'], video['title']}")

print(f'All transcripts collected for the channel: "{channel_id}"')