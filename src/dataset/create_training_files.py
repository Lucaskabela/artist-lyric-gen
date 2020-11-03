import json
import datetime
import time
import re
from tqdm import tqdm
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file
from apply_bpe_to_songs import bpe_songs_dir

def create_train_file(list_file, out_file):
    verses_list = []
    song_names = read_list_from_file(list_file)
    for song_name in tqdm(song_names):
        with open("{}/{}.json".format(bpe_songs_dir, name_to_file_name(song_name))) as song_file:
            song = json.load(song_file)
        for verse in song['verses']:
            if verse['valid']:
                train_verse = {}
                train_verse['artist_id'] = verse['artist_id']
                train_verse['lyrics'] = verse['lyrics']
                verses_list.append(train_verse)
    with open(out_file, 'w') as openfile:
        json.dump(verses_list, openfile)

