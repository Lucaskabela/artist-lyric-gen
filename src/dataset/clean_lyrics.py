import json
import re
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file
from tqdm import tqdm
from get_songs import raw_songs_dir

cleaned_songs_dir = "cleaned_raw_lyrics"

def clean_lyrics(s):
    cleaned_lyrics = s.lower()
    # Do the cleaning steps
    # Removes (...), {...}, * ... *
    # Removes [?]
    cleaned_lyrics = re.sub(r'\[\?\]', ' ', cleaned_lyrics)
    #TODO: Add more cleaning
    # Removes extra spaces
    cleaned_lyrics = re.sub(r' +', ' ', cleaned_lyrics)
    # Removes extra lines, and strips lines
    lines = [line.strip() for line in cleaned_lyrics.split('\n')]
    lines = list(filter(lambda s: s != '', lines))
    # reconstruct the lines together
    cleaned_lyrics = ''
    for line in lines:
        # concat lines together and add the end line token back
        cleaned_lyrics = cleaned_lyrics + line + '\n'
    return cleaned_lyrics

def process_song(song, bar):
    with open("{}/{}.json".format(raw_songs_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    song['lyrics'] = clean_lyrics(song['lyrics'])
    return song


def clean_song(song_list_path, out_dir):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", lambda x:x, out_dir)