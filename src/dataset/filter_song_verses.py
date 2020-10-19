from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import json
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file
from raw_song_to_verse_split_song_files import verse_split_songs_dir

filtered_verses_dir = "filtered_verses"

def filter_song(song):
    for verse in song['verses']:
        if detect(verse['lyrics']) != 'en':
            return True
    return False

def process_song(song):
    with open("{}/{}.json".format(verse_split_songs_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    if filter_song(song):
        return False
    return None

def filter_verse_songs(song_list_path, out_dir_prefix):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", lambda x:x , out_dir_prefix)
