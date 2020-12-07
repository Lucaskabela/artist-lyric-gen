import json
import datetime
import time
import re
from tqdm import tqdm
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file
from fix_tokens import verses_with_tokens


def add_file_contents(out_file, input_file):
    with open(input_file) as infile:
        out_file.write(infile.read() + "\n")

def write_out_songs(song_names, outfile):
    for song_name in tqdm(song_names):
        with open("{}/{}.json".format(verses_with_tokens, name_to_file_name(song_name))) as song_file:
            song = json.load(song_file)
        for verse in song['verses']:
            if verse['valid']:
                outfile.write(verse['lyrics'] + '\n')

def compile_lyrics(list_file, out_file):
    with open(out_file, 'w') as outfile:
        outfile.write('W\n')
        add_file_contents(outfile, 'personas_tags_fixed.txt')
        add_file_contents(outfile, 'personas_sentences_clean.txt')
        song_names = read_list_from_file(list_file)
        write_out_songs(song_names, outfile)

