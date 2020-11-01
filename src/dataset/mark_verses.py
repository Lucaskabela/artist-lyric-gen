import json
import re
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file
from verse_songs_extract_artists import verse_artists_dir

marked_verses_dir = "marked_verses"
skipped_artists_file = "_SKIPPED_ARTISTS"

def is_verse_artist_valid(verse, artist_list):
    def write_out_artist(artist):
        with open("{}/{}".format(marked_verses_dir,skipped_artists_file), 'a') as openfile:
            openfile.write('{}\n'.format(artist))
    if len(verse['artists']) != 1:
        # only allow 1 artist
        return False
    elif verse['artists'][0] not in artist_list:
        # write out the artists we skip, can verify if we need to
        # clean any artist names
        write_out_artist(verse['artists'][0])
        return False
    else:
        return True

def is_verse_type_valid(verse):
    return re.search("skit|sample", verse['metadata'].lower()) is None

def has_enough_lines(verse):
    # want verses with >= 4 lines
    # \n at the end of each line
    # \n at the end of the whole verse
    return len(re.findall('\n', verse['lyrics'])) >= 5

def get_process_song(artist_list):
    def process_song(song, bar):
        with open("{}/{}.json".format(verse_artists_dir, name_to_file_name(song))) as song_file:
            song = json.load(song_file)
        for verse in song['verses']:
            verse['valid'] = is_verse_artist_valid(verse, artist_list) and \
                is_verse_type_valid(verse) and \
                has_enough_lines(verse)
        for verse in song['verses']:
            if verse['valid']:
                return song
        # if all verses are invalid, remove the song
        return False
    return process_song

def mark_verses(song_list_path):
    song_list = read_list_from_file(song_list_path)
    artist_list = read_list_from_file('cleaned_artist_names.txt')
    process_song = get_process_song(artist_list)
    loop_and_process(song_list, process_song, "Song", lambda x: x, marked_verses_dir)
    remove_duplicates_from_list_file(marked_verses_dir, skipped_artists_file)
