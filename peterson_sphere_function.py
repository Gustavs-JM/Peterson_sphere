###

"""
1 Load the libraries
2 Load the csv of channel names
3 Load the list of video directories for the specified channel name
3 Use the API to get the data from the channel, compile a list of videos from the channel
"""

"""
Function to fetch all videos from a YouTube channel.
Requires:
- Google API Client: pip install google-api-python-client
"""
#
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
import json
import os
import requests
import time
import yaml
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp

def get_channel_info(api_key, channel_id=None, channel_username=None):
    """
    Get information about a YouTube channel using either channel ID or username.

    Args:
        api_key (str): Your YouTube Data API key
        channel_id (str, optional): YouTube channel ID (starts with UC...)
        channel_username (str, optional): YouTube channel username/handle

    Returns:
        dict: Channel information
    """
    base_url = "https://www.googleapis.com/youtube/v3/channels"

    # Determine which parameter to use for identifying the channel
    if channel_id:
        params = {
            'key': api_key,
            'id': channel_id,
            'part': 'snippet,contentDetails,statistics'
        }
    elif channel_username:
        params = {
            'key': api_key,
            'forUsername': channel_username,
            'part': 'snippet,contentDetails,statistics'
        }
    else:
        raise ValueError("Either channel_id or channel_username must be provided")

    # Make the API request
    response = requests.get(base_url, params=params)
    print(response)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # print(data)

        # Check if any channels were found
        if data['pageInfo']['totalResults'] > 0:
            channel = data['items'][0]
            print(channel)

            # Extract relevant information
            channel_info = {
                'id': channel['id'],
                'title': channel['snippet']['title'],
                'description': channel['snippet']['description'],
                'customUrl': channel['snippet'].get('customUrl', 'Not available'),
                'publishedAt': channel['snippet']['publishedAt'],
                'thumbnails': channel['snippet']['thumbnails'],
                'subscriberCount': channel['statistics'].get('subscriberCount', 'Not available'),
                'videoCount': channel['statistics'].get('videoCount', 'Not available'),
                'viewCount': channel['statistics'].get('viewCount', 'Not available'),
                'uploadsPlaylistId': channel['contentDetails']['relatedPlaylists']['uploads']
            }

            return channel_info
        else:
            return {'error': 'No channel found with the provided ID or username'}
    else:
        return {'error': f'API request failed with status code {response.status_code}', 'details': response.text}

def get_recent_videos(api_key, uploads_playlist_id, max_results=10):
    """
    Get the most recent videos from a channel using the uploads playlist ID.

    Args:
        api_key (str): Your YouTube Data API key
        uploads_playlist_id (str): The ID of the uploads playlist
        max_results (int, optional): Maximum number of videos to retrieve

    Returns:
        list: List of recent videos
    """
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"

    params = {
        'key': api_key,
        'playlistId': uploads_playlist_id,
        'part': 'snippet,contentDetails',
        'maxResults': max_results
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        videos = []
        for item in data['items']:
            video = {
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'publishedAt': item['snippet']['publishedAt'],
                'videoId': item['contentDetails']['videoId'],
                'thumbnails': item['snippet']['thumbnails'],
                'channel': item['snippet']['channelTitle']
            }
            videos.append(video)

        return videos
    else:
        return {'error': f'API request failed with status code {response.status_code}', 'details': response.text}

def get_all_channel_videos(api_key, uploads_playlist_id, max_videos=None):
    """
    Get all videos from a channel's uploads playlist with pagination.

    Args:
        api_key (str): Your YouTube Data API key
        uploads_playlist_id (str): The ID of the uploads playlist
        max_videos (int, optional): Maximum number of videos to retrieve (None for all)

    Returns:
        list: Complete list of videos
    """
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"

    videos = []
    next_page_token = None
    page_counter = 0

    while True:
        # Add delay to avoid rate limiting
        if page_counter > 0:
            time.sleep(random.uniform(40, 500))  # 40-500 second delay between requests

        # Parameters for the API request
        params = {
            'key': api_key,
            'playlistId': uploads_playlist_id,
            'part': 'snippet,contentDetails',
            'maxResults': 50  # Max allowed by the API
        }

        # Add the pageToken if we have one
        if next_page_token:
            params['pageToken'] = next_page_token

        # Make the API request
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            return {'error': f'API request failed with status code {response.status_code}', 'details': response.text}

        data = response.json()

        # Process the videos in the current page
        for item in data['items']:
            video = {
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'publishedAt': item['snippet']['publishedAt'],
                'videoId': item['contentDetails']['videoId'],
                'thumbnails': item['snippet']['thumbnails'],
                'channel': item['snippet']['channelTitle']
            }
            videos.append(video)

            # If we've reached the maximum requested videos, stop
            if max_videos and len(videos) >= max_videos:
                return videos[:max_videos]

        # Check if there are more pages
        next_page_token = data.get('nextPageToken')
        page_counter += 1

        # If no more pages or we've reached our limit, break the loop
        if not next_page_token:
            break

        # Optional: Print progress
        print(f"Retrieved {len(videos)} videos so far. Getting the next page...")

    return videos

def get_video_transcript(video_id):
    """
    Get the transcript for a YouTube video using the youtube_transcript_api.

    Args:
        video_id (str): YouTube video ID

    Returns:
        dict: A dictionary with 'success' (bool), 'transcript' (list/str) and 'error' (str) if applicable
    """
    time.sleep(random.uniform(40, 500)) # Time randomisation to avoid getting blocked

    try:

        transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en'])

        return {
            'success': True,
            'transcript': transcript_data,  # Original data with timestamps
            'language': 'en'
        }

    except TranscriptsDisabled:
        return {
            'success': False,
            'error': 'Transcripts are disabled for this video'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def save_transcript_to_yaml(video_id, video_info=None):
    """
    Retrieve and save a YouTube transcript to a YAML file with proper formatting.

    Args:
        video_id (str): YouTube video ID
        video_info (dict, optional): Additional video metadata
        output_dir (str): Directory to save the YAML file

    Returns:
        dict: Result of the operation with file path if successful
    """

    # Get the transcript
    result = get_video_transcript(video_id)

    if not result['success']:
        return {
            'success': False,
            'video_id': video_id,
            'error': result.get('error', 'Failed to retrieve transcript')
        }

    # Format the transcript data
    transcript_data = result['transcript']

    # Prepare YAML data structure
    yaml_data = {
        'metadata': {
            'video_id': video_id,
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'transcript': []
    }

    # Add additional video metadata if provided
    if video_info:
        yaml_data['metadata'].update({
            'title': video_info.get('title', 'Unknown'),
            'channel': video_info.get('channel', 'Unknown'),
            'published_at': video_info.get('publishedAt', 'Unknown'),
            'duration': video_info.get('duration', 'Unknown'),
            'participants': video_info.get('participants', 'Unknown'),
            'keywords': video_info.get('keywords', 'Unknown'),
            'views': video_info.get('views', 'Unknown')
        })

    # Format transcript entries
    for entry in transcript_data:
        formatted_entry = {
            'start': entry.start,
            ##'start_formatted': format_time(entry.get('start', 0)),
            ##'duration': entry.get('duration', 0),
            'text': entry.text
        }
        yaml_data['transcript'].append(formatted_entry)

    return yaml_data

def save_whisperx_transcript_to_yaml(result, output_file, audio_file):
    """
    Save WhisperX transcript result to YAML format.

    Args:
        result: WhisperX transcription result dictionary
        output_file: Output YAML file path

    Returns:
        Path to saved YAML file
    """

    # Prepare data for YAML
    transcript_data = {
        'metadata': {
            'audio_file': str(audio_file),
            'language': result.get('language', 'unknown'),
            'transcription_date': datetime.now().isoformat(),
            'total_segments': len(result.get('segments', [])),
            'model_used': 'whisperx'
        },
        'full_text': result.get('full_text', ''),
        'segments': []
    }

    # Process segments
    for i, segment in enumerate(result.get('segments', [])):
        segment_data = {
            'id': i + 1,
            'start_time': round(segment.get('start', 0), 2),
            'end_time': round(segment.get('end', 0), 2),
            'text': segment.get('text', '').strip(),
            'speaker': segment.get('speaker', 'Unknown')
        }

        # Add word-level data if available
        if 'words' in segment:
            segment_data['words'] = []
            for word in segment['words']:
                word_data = {
                    'word': word.get('word', ''),
                    'start': round(word.get('start', 0), 2),
                    'end': round(word.get('end', 0), 2),
                    'confidence': round(word.get('score', 0), 3) if 'score' in word else None
                }
                segment_data['words'].append(word_data)

        transcript_data['segments'].append(segment_data)

    # Save to YAML file
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(transcript_data, f,
                  default_flow_style=False,
                  allow_unicode=True,
                  indent=2,
                  sort_keys=False)

    return str(output_file)


def make_video_filename(video_info=None):
    """
    :param video_info: various points of info about the video (dict)
    :return: channel name and filename for video-associated directory (dict)
    """

    invalid_chars = list(r'[<>:"/\\|?*]')
    if video_info is not None:
        publishedAt = video_info['publishedAt']
        videoId = video_info['videoId']
        title = video_info['title']

        channel_name = video_info['channel']

        new_time = publishedAt.replace('-', '_')[:10]
        for c in invalid_chars:
            title = title.replace(' ', '').replace(c, '')
        new_title = title[:20]

        new_filename = new_time + '_' + videoId + '_' + new_title
        return {'channel': channel_name, 'filename': new_filename}

    else:
        print('error')

def get_video_id_from_filename(filename):
    """
    :param filename: Takes the name of a file where metadata or transcripts are saved
    :return: the associated youtube video id
    """
    video_id = filename[11:22]
    return video_id

def guarantee_directories(database_name, channel_id, video_filename):
    """
    Makes the necessary folders if they don't exist

    :param database_name: root folder of the database (str)
    :param channel_id: channelId of the channel the video is from (str)
    :param video_filename: generated name for the directory associated with this video (str)
    :return: the path to the video-associated folder
    """
    os.makedirs(database_name, exist_ok=True)

    channel_folder_name = os.path.join(database_name, channel_id)
    os.makedirs(channel_folder_name, exist_ok=True)

    print(channel_folder_name, video_filename['filename'])
    video_folder_name = os.path.join(channel_folder_name, video_filename['filename'])
    os.makedirs(video_folder_name, exist_ok=True)

    return video_folder_name

def save_yaml_to_address(video_filename, filename, yaml_file):
    """
    Saves a yaml file into the desired location
    :param video_filename: the name of the folder associated with the video
    :param filename: the specific name of the file, transcript or metadata etc
    :param yaml_file: the contents of the yaml file that is to be saved
    :return:
    """

    file_path = os.path.join(video_filename, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Write YAML file
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_file, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return file_path

def get_list_of_saved_local_transcripts(database_name, channel_name):
    """
    Finds out which transcripts have been already saved to avoid calling for their transcripts
    :param database_name: the root of the database (str)
    :param channel_id: the channel to search through (str)
    :return: a dictionary of videoIds and types of files created associated with them
    """

    analysis_path = os.path.join(database_name, channel_name)
    try:
        contents = os.listdir(analysis_path)

        paths = [os.path.join(analysis_path, x) for x in contents]

        files = {get_video_id_from_filename(x):[y.split('_', 1)[0] for y in os.listdir(os.path.join(analysis_path, x))] for x in contents}

        return files
    except :
        print(f'ERROR: Database "{database_name}" not found: {database_name}')
        return {}


def extract_audio(youtube_url, output_path, filename):

    time.sleep(random.uniform(40, 500))

    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'audioquality': '128k',  # Adjust quality here
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': f'{output_path}/{filename}.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])