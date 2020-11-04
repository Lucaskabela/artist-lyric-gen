import json
import re
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file, clean_artist_names
from tqdm import tqdm
from mark_verses import marked_verses_dir

verses_with_tokens = "verses_with_tokens"

def process_song(song, bar):
    with open("{}/{}.json".format(marked_verses_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    for verse in song['verses']:
        new_lyrics = verse['lyrics']
        lines = [line.strip() for line in new_lyrics.split('\n')]
        lines = list(filter(lambda s: s != '', lines))
        # reconstruct the lines together
        cleaned_lyrics = ''
        for line in lines:
            # concat lines together and add the end line token back
            cleaned_lyrics = cleaned_lyrics + 'S ' + line + ' L '
        cleaned_lyrics = cleaned_lyrics.strip()
        verse['lyrics'] = cleaned_lyrics
    return song


def fix_tokens_for_verses(song_list_path, out_dir):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", lambda x:x, out_dir)