from langdetect import detect, DetectorFactory, lang_detect_exception
DetectorFactory.seed = 0
import json
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file
from get_songs import raw_songs_dir

filtered_songs_dir = "filtered_songs"

def filter_song(song):
    try:
        lang = detect(song['lyrics'])
    except:
        print("{} has no features".format(song['title']))
        return True
    if lang != 'en':
        print("{} is {}".format(song['title'], lang))
        return True
    return False

def process_song(song):
    with open("{}/{}.json".format(raw_songs_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    if filter_song(song):
        return False
    return None

def filter_songs(song_list_path, out_dir_prefix):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", lambda x:x , out_dir_prefix)
