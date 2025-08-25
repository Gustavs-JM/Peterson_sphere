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

torchaudio.set_audio_backend("ffmpeg")  # or try "sox"

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
            sleep_length = random.uniform(5, 30)
            print(f'Waiting for {sleep_length} seconds before making the next request')
            time.sleep(sleep_length)  # 40-500 second delay between requests

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
            'start': segment.get('start', 0),
            'end': segment.get('end', 0),
            'text': segment.get('text', '').strip(),
            'speaker': segment.get('speaker', 'Unknown')
        }

        transcript_data['segments'].append(segment_data)

    # Save to YAML file
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(transcript_data, f,
                  default_flow_style=False,
                  allow_unicode=True,
                  indent=2,
                  sort_keys=False)
    return transcript_data


def get_foldername_from_video_id(database_name, channel_name, video_id):
    channel_folder_address = database_name + '/' + channel_name
    foldername_list = os.listdir(channel_folder_address)
    splitted_foldernames = {x[11:22]:x for x in foldername_list}
    return splitted_foldernames.get(video_id, 0)

def make_whisper_transcript(audiofile_address, min_speakers, max_speakers, HF_TOKEN, whisperx_transcript_address, video_id):
    try:
        result = transcribe_basic(
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

    wait_time = random.uniform(20, 240)
    print(f'Waiting for {wait_time} seconds before making the next audio download request')
    time.sleep(wait_time)

    print(f'Output path: {output_path}, filename: {filename}')

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



def get_youtube_video_data_from_id(video_id, api_key):

    """
    Fetch video data from YouTube API using video ID

    Args:
        video_id (str): YouTube video ID (e.g., 'dQw4w9WgXcQ')
        api_key (str): Your YouTube Data API v3 key

    Returns:
        dict: Video data including title, description, channel info, etc.
    """

    # YouTube API endpoint
    url = "https://www.googleapis.com/youtube/v3/videos"

    # Parameters for the API call
    params = {
        'part': 'snippet,statistics,contentDetails',
        'id': video_id,
        'key': api_key
    }

    try:
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Check if video was found
        if not data.get('items'):
            return {'error': 'Video not found or private/deleted'}

        # Extract video info
        video_info = data['items'][0]
        snippet = video_info['snippet']
        statistics = video_info.get('statistics', {})
        content_details = video_info.get('contentDetails', {})

        # Return structured data
        return {
            'video_id': video_id,
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'channel_title': snippet.get('channelTitle', ''),
            'channel_id': snippet.get('channelId', ''),
            'published_at': snippet.get('publishedAt', ''),
            'duration': content_details.get('duration', ''),
            'view_count': statistics.get('viewCount', '0'),
            'like_count': statistics.get('likeCount', '0'),
            'comment_count': statistics.get('commentCount', '0'),
            'tags': snippet.get('tags', []),
            'category_id': snippet.get('categoryId', ''),
            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
            'raw_data': video_info  # Full API response if needed
        }

    except requests.exceptions.RequestException as e:
        return {'error': f'API request failed: {str(e)}'}
    except json.JSONDecodeError as e:
        return {'error': f'Failed to parse response: {str(e)}'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}



def claude_initial_description_analysis(title, description, tags, claude_api_key):
    """
        Extract list of people mentioned in a YouTube video using Claude API

        Args:
            title (str): YouTube video title
            description (str): YouTube video description
            api_key (str): Your Anthropic API key

        Returns:
            list: List of people names mentioned in the video
        """
    # Claude API endpoint
    url = "https://api.anthropic.com/v1/messages"

    # Headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": claude_api_key,
        "anthropic-version": "2023-06-01"
    }

    # Prompt for extracting people
    prompt = f"""
        Based on this YouTube video title, description and tags, please identify all people who are likely talking in this video, not just people who are referred to. Return only a JSON list of names.

        Title: {title}
        Description: {description}
        Tags: {tags}

        Please return ONLY a valid JSON array of names, like: ["Name 1", "Name 2", "Name 3"]
        If no specific people are mentioned, return an empty array: []
        """

    # Request payload
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        # Parse response
        result = response.json()
        content = result['content'][0]['text'].strip()

        # Try to parse JSON response
        try:
            people_list = json.loads(content)
            return people_list
        except json.JSONDecodeError:
            # If Claude didn't return valid JSON, try to extract names manually
            print(f"Warning: Could not parse JSON response: {content}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        return []

def generic_claude_response(prompt, claude_api_key):
    """
        Extract list of people mentioned in a YouTube video using Claude API

        Args:
            title (str): YouTube video title
            description (str): YouTube video description
            api_key (str): Your Anthropic API key

        Returns:
            list: List of people names mentioned in the video
        """
    # Claude API endpoint
    url = "https://api.anthropic.com/v1/messages"

    # Headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": claude_api_key,
        "anthropic-version": "2023-06-01"
    }

    # Request payload
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        # Parse response
        result = response.json()
        content = result['content'][0]['text'].strip()

        # Try to parse JSON response
        try:
            people_list = json.loads(content)
            return people_list
        except json.JSONDecodeError:
            # If Claude didn't return valid JSON, try to extract names manually
            print(f"Warning: Could not parse JSON response: {content}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        return []

def compare_audio_files(audio_clip, voice_samples_directory):
    # Load pre-trained model
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

    main_waveform, _ = librosa.load(audio_clip, sr=16000)
    main_waveform = torch.tensor(main_waveform).unsqueeze(0)

    results = {}
    voice_dir = Path(voice_samples_directory)

    for voice_file in voice_dir.iterdir():
        if voice_file.suffix.lower() in ['.wav', '.mp3']:
            try:
                # Load comparison audio
                comp_waveform, _ = librosa.load(voice_file, sr=16000)
                comp_waveform = torch.tensor(comp_waveform).unsqueeze(0)

                # Compare using verify_batch
                score, prediction = verification.verify_batch(main_waveform, comp_waveform)

                results[voice_file.name] = {
                    'score': float(score),
                    'same_speaker': bool(prediction)
                }

            except Exception as e:
                print(f"Error processing {voice_file.name}: {e}")
                continue

    return results


def restructure_whisperx_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restructure WhisperX result into a cleaner format for YAML output.

    Args:
        result: The original WhisperX result dictionary

    Returns:
        Restructured dictionary with simplified format
    """

    restructured = {
        "metadata": {
            "language": result.get("language", "unknown"),
            "language_probability": result.get("language_probability", 0.0),
            "total_segments": len(result.get("segments", [])),
            "total_duration": 0.0
        },
        "segments": []
    }

    # Calculate total duration
    if result.get("segments"):
        last_segment = result["segments"][-1]
        restructured["metadata"]["total_duration"] = last_segment.get("end", 0.0)

    # Process segments
    for i, segment in enumerate(result.get("segments", []), 1):
        clean_segment = {
            "segment_number": i,
            "speaker": segment.get("speaker", "Unknown"),
            "start_time": float(segment.get("start", 0)),
            "end_time": float(segment.get("end", 0)),
            "text": segment.get("text", "").strip()
        }

        restructured["segments"].append(clean_segment)

    return restructured


def extract_segment_librosa(audio_path, start_time, end_time, sr=16000):
    """Extract audio segment using librosa"""

    # Load full audio
    audio, original_sr = librosa.load(audio_path, sr=sr)

    # Convert time to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract segment
    segment = audio[start_sample:end_sample]

    # Convert to torch tensor with batch dimension
    segment_tensor = torch.tensor(segment).unsqueeze(0).float()

    return segment_tensor, sr

def compare_segment_with_file(main_audio_path, start_time, end_time, comparison_audio_path):
    """Compare audio segment with another file using tensors"""

    # Extract segment from main audio
    segment_tensor, sr = extract_segment_librosa(main_audio_path, start_time, end_time)

    # Load comparison audio
    comparison_audio, _ = librosa.load(comparison_audio_path, sr=sr)
    comparison_tensor = torch.tensor(comparison_audio).unsqueeze(0).float()

    # Load SpeechBrain model
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

    # Compare using verify_batch
    score, prediction = verification.verify_batch(segment_tensor, comparison_tensor)

    return {
        'score': float(score),
        'same_speaker': bool(prediction),
        'segment_duration': len(segment_tensor[0]) / sr
    }

def assign_speakers_through_audio_comparison(main_audio_path, yaml_transcript, voice_sample_directory):

    assigned_names = {}
    speakers = set()
    segments = yaml_transcript.get("segments", [])
    for segment in segments:
        speaker = segment.get('speaker')
        if speaker:
            speakers.add(speaker)
    speakers = list(speakers)
    print(speakers)

    for speaker in speakers:
        if speaker[:7] == 'SPEAKER':
            ## Find the segment start and end for voice comparison
            print(speaker)
            speaker_length_list = []
            sorted_list = []
            sorted_list_2 = []
            for segment in segments:
                if segment['speaker']==speaker:
                    length = segment['end']-segment['start']
                    speaker_length_list.append([length, segment['start'], segment['end']])
            for x in speaker_length_list:
                if x[0]>10:
                    sorted_list.insert(0, x)
                else:
                    sorted_list.append(x)
            for x in sorted_list:
                if x[0] < 25:
                    sorted_list_2.insert(0, x)
                else:
                    sorted_list_2.append(x)
            chosen_segment = sorted_list_2[0]
            chosen_start = chosen_segment[1]
            chosen_end = chosen_segment[2]
            if chosen_end - chosen_start > 16:
                chosen_end = chosen_start + 16

            comparisons_with_samples = {}
            # Load the segment and compare it with each speaker sample
            voice_dir = Path(voice_sample_directory)
            for voice_file in voice_dir.iterdir():
                if voice_file.suffix.lower() in ['.wav', '.mp3']:
                    one_comparison = compare_segment_with_file(
                        main_audio_path=main_audio_path,
                        start_time=chosen_start,
                        end_time=chosen_end,
                        comparison_audio_path=voice_file
                    )
                    voice_name, ext = os.path.splitext(os.path.basename(voice_file))
                    comparisons_with_samples[voice_name] = one_comparison

            good_comparisons = {}
            for name in comparisons_with_samples:
                if comparisons_with_samples[name]['same_speaker'] == True:
                    good_comparisons[name] = comparisons_with_samples[name]['score']
            if len(good_comparisons)>0:
                detected_name = max(good_comparisons, key=good_comparisons.get)
                assigned_names[speaker] = detected_name
        else:
            pass

    print(assigned_names)
    new_segments = []
    for segment in segments:
        speaker = segment.get('speaker')
        if speaker in assigned_names:
            replacement_name = assigned_names[speaker]
            segment['speaker'] = replacement_name
            new_segments.append(segment)
        else:
            new_segments.append(segment)
    new_yaml = yaml_transcript
    new_yaml['segments'] = new_segments

    for x in new_segments:
        print(x)
    return new_yaml

def shorten_transcript(yaml):

    segments = yaml.get('segments', [])
    grouped_indexes = []

    current_group = [0]
    for i, segment in enumerate(segments[:-1]):
        current_speaker = segment.get('speaker', '')
        next_speaker = segments[i+1].get('speaker', '')
        if current_speaker == next_speaker:
            current_group.append(i+1)
        else:
            grouped_indexes.append(current_group)
            current_group = [i+1]
    grouped_indexes.append(current_group)


    new_transcript = []
    for i, group in enumerate(grouped_indexes):
        group_texts = [segments[x].get('text', '') for x in group]
        group_segment = {
            'id': i+1,
            'speaker': segments[group[0]].get('speaker', 'Unknown'),
            'text': ' '.join(group_texts),
            'start': segments[group[0]].get('start'),
            'end': segments[group[-1]].get('end')
        }
        new_transcript.append(group_segment)

    yaml['segments'] = new_transcript

    return yaml

def assign_speakers_through_api_judgement(
        yaml_transcript,
        speech_clip_directory,
        video_id,
        CLAUDE_API_KEY,
        YOUTUBE_TOKEN):
    a=1
    while a==1:
        print('starting the while loop')
        speech_segments = yaml_transcript.get('segments')
        print(speech_segments)
        speakers = list(set([x['speaker'] for x in speech_segments]))
        print(speakers)
        unknown_speakers = [x for x in speakers if x[:7] == 'SPEAKER']
        known_speakers = [x for x in speakers if x[:7] != 'SPEAKER']
        if len(unknown_speakers) == 0:
            break

        speaker = unknown_speakers[0]
        print(f'Starting the loop with speaker: {speaker}')

        prompt_header = """"
        You have the task of determining the identity of an unidentified speaker in a transcript of a youtube video. You will have to make a judgement, using the information given to you in this prompt. Try hard to give a name. If nothing is possible, return a name in the format of 'Unknown_1' or 'Unknown_2' etc., using a name that is not already used.
        Return as an answer a name in a JSON format ONLY!, such as, for example, ['Jordan Peterson']. DO NOT RETURN MORE COMPLEX REPLIES, ONLY A SINGLE NAME!"""

        ## Get the text samples from the target speaker
        speaker_samples = [x['text'][:1000] for x in speech_segments if x['speaker']==speaker][:3]
        if speaker_samples:
            input_speaker_samples = f"""
            These are samples from what the unknown speaker said. It should help you make a judgement on their identity: {speaker_samples}
            """
        else:
            input_speaker_samples = ''


        ## Get the speaker samples from the preceding text samples.
        speaker_sample_indexes = [x['id'] for x in speech_segments if x['speaker']==speaker][:3]
        preceding_ids = [x-1 for x in speaker_sample_indexes]
        preceding_ids_2 = []
        for i in preceding_ids:
            if i > 1:
                preceding_ids_2.append(i-2)
        if preceding_ids_2:
            preceding_texts = {speech_segments[x]['speaker']: speech_segments[x]['text'][-150:] for x in preceding_ids_2}
        if preceding_texts:
            input_preceding_samples = f""""
            These are samples from things people in conversation, given with names of the speakers (which may or may not be known) said directly before the speaker you need to identify spoke. They may offer clues into the target speaker's identity: {preceding_texts}
            """
        else:
            input_preceding_samples = ''


        ### Get names of people that the target speaker is NOT:
        voice_dir = Path(speech_clip_directory)
        excluded_names = []
        for voice_file in voice_dir.iterdir():
            voice_name, ext = os.path.splitext(os.path.basename(voice_file))
            excluded_names.append(voice_name)
        if excluded_names:
            input_excluded_names = f""""
            These are names of people that the target speaker is NOT: {excluded_names}
            """
        else:
            input_excluded_names= ''

        print('Making the known_speakers')
        ### Get known names in the video:
        if known_speakers:
            input_known_interlocutors = f"""
            These are people who have been determined to also be in this conversation, but are NOT the target speaker: {[x for x in known_speakers if x != 'Unknown']}
            """
        else:
            input_known_interlocutors = ''


        ## Get the YOUTUBE data
        video_data = get_youtube_video_data_from_id(video_id, YOUTUBE_TOKEN)

        video_title = video_data.get('title')
        video_description = video_data.get('description')[:2000]
        video_tags = video_data.get('tags')

        input_video_data = f""""
        This is the title of the video this converstation is in: {video_title}
        This is the first part of the description of the video: {video_description}
        These are the tags of the video: {video_tags}
        """

        summarised_prompt = prompt_header + input_speaker_samples + input_preceding_samples + input_excluded_names + input_known_interlocutors + input_video_data


        #print(summarised_prompt)
        print('Getting the claude response now:')
        claude_response = generic_claude_response(summarised_prompt, CLAUDE_API_KEY)[0]
        print('CLAUDE RESPONSE: ', claude_response)

        new_segments = []
        for segment in speech_segments:
            id_speaker = segment.get('speaker')
            if id_speaker == speaker:
                replacement_name = claude_response
                segment['speaker'] = replacement_name
                new_segments.append(segment)
            else:
                new_segments.append(segment)
        new_yaml = yaml_transcript
        new_yaml['segments'] = new_segments

        yaml_transcript = new_yaml

        print(yaml_transcript)

        speech_segments = yaml_transcript.get('segments')
        print(speech_segments)
        speakers = list(set([x['speaker'] for x in speech_segments]))
        print(speakers)
        unknown_speakers = [x for x in speakers if x[:7] == 'SPEAKER']
        known_speakers = [x for x in speakers if x[:7] != 'SPEAKER']
        if len(unknown_speakers) == 0:
            a=2
    print(yaml_transcript)


    return yaml_transcript
