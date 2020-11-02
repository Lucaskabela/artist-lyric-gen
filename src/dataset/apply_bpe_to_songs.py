from subword_nmt import apply_bpe
import json
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file, clean_artist_names, get_bpe_object, apply_bpe_to_string
from fix_tokens import verses_with_tokens

bpe_songs_dir = 'bpe_songs'

def run_bpe_on_songs(codes_file, song_list_path, out_dir):
    bpe = get_bpe_object(codes_file)
    song_list = read_list_from_file(song_list_path)
    def process_song(song, bar):
        with open("{}/{}.json".format(verses_with_tokens, name_to_file_name(song))) as song_file:
            song = json.load(song_file)
        for verse in song['verses']:
            if verse['valid']:
                lyrics = verse['lyrics']
                lyrics = apply_bpe_to_string(lyrics, bpe)
                verse['lyrics'] = lyrics
        return song
    loop_and_process(song_list, process_song, "Song", lambda x:x, out_dir)

