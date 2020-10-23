from langdetect import detect, DetectorFactory, lang_detect_exception
DetectorFactory.seed = 0
import json
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file
from clean_verses import cleaned_verses_dir

filtered_songs_dir = "filtered_songs"

def filter_lyrics(lyrics, title, bar):
    try:
        lang = detect(lyrics.strip())
    except:
        bar.write("IIII {} has no features".format(title))
        return True
    if lang != 'en':
        bar.write("OOOO {} has {}".format(title, lang))
        return True
    return False


def filter_song(song, bar):
    if len(song['verses']) == 0:
        bar.write("XXXX {} has no verses".format(song['title']))
        return True
    if len(song['verses']) > 15:
        bar.write("XXXX {} has too many verses".format(song['title']))
        return True
    all_lyrics = ''
    for verse in song['verses']:
        all_lyrics = all_lyrics + verse['lyrics']
    # cant use a song with another language in it
    if filter_lyrics(all_lyrics, song['title'], bar):
        return True
    return False


def process_song(song, bar):
    with open("{}/{}.json".format(cleaned_verses_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    if filter_song(song, bar):
        return False
    return None

def filter_songs(song_list_path, out_dir_prefix):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", lambda x:x , out_dir_prefix)
