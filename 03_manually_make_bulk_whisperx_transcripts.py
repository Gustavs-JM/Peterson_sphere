from peterson_sphere_function import *
import yt_dlp
from whisperx_transcription import transcribe_basic

"""
This script is meant to be called through the terminal
with manual input of the channel id
to find all the audio files saved for that channel
and make diarized whisperx transcripts for those files
"""


keys = open('keys.txt')
key_list = keys.readlines()
API_KEY = key_list[0]

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
print(channel_name)

### Find which videos have saved audio files
existing_saved_files = get_list_of_saved_local_transcripts(database_name, channel_name)
existing_saved_audios = [x for x in existing_saved_files if 'audio' in existing_saved_files.get(x, []) ]
existing_saved_whisperx_transcripts = [x for x in existing_saved_files if 'whisperx' in existing_saved_files.get(x, []) ]
existing_audio_but_no_transcript = [x for x in existing_saved_audios if x not in existing_saved_whisperx_transcripts]
print(existing_audio_but_no_transcript)


### Make a list of file paths for the different audio files
def get_foldername_from_video_id(database_name, channel_name, video_id):
    channel_folder_address = database_name + '/' + channel_name
    foldername_list = os.listdir(channel_folder_address)
    splitted_foldernames = {x[11:22]:x for x in foldername_list}
    return splitted_foldernames.get(video_id, 0)


get_foldername_from_video_id(database_name, channel_name, 1)
for video_id in existing_audio_but_no_transcript:

    ## Get the address of the audio file in the folder
    folder_name = get_foldername_from_video_id(database_name, channel_name, video_id)
    folder_address = database_name+'/'+channel_name+'/'+folder_name
    full_filelist = os.listdir(folder_address)
    audiofilename = [x for x in full_filelist if x.split('_')[0]=='audio'][0]
    audiofile_address = folder_address+'/'+audiofilename
    print(audiofile_address)

    ## Send it to the transcription script

    whisperx_transcript_address = folder_address + '/whisperx_' + folder_name+ '.yaml'

    if not os.path.exists(audiofile_address):
        print(f"File not found: {audiofile_address}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
    try:
        result = transcribe_basic(
            audio_file=audiofile_address,
            model_size="tiny",  # or "medium", "small", etc.
            device="cpu",  # or "cpu" if no GPU, original "cuda"
            compute_type="int8", # This exists because I was running on my small device
        )

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

                print(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}")

            save_whisperx_transcript_to_yaml(result, whisperx_transcript_address, video_id)
        else:
            print("Transcription failed!")

    except Exception as e:
        print(f"Error: {e}")



exit()



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