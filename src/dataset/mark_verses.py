import json
import re
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file, clean_artist_names
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
    return re.search(r"skit|sample", verse['metadata'].lower()) is None

def has_enough_lines(verse):
    # want verses with >= 4 lines
    # \n at the end of each line
    return len(re.findall('\n', verse['lyrics'])) >= 4

def get_process_song(artist_list):
    def process_song(song, bar):
        with open("{}/{}.json".format(verse_artists_dir, name_to_file_name(song))) as song_file:
            song = json.load(song_file)
        # TODO: we just add this here for ease, but it should be moved
        # somewhere else
        # here we remove songs that are not by someone in our artist list
        if clean_artist_names(song['artist']).strip() not in artist_list:
            return False
        verse_lyrics_set = set()
        for verse in song['verses']:
            verse['valid'] = is_verse_artist_valid(verse, artist_list) and \
                is_verse_type_valid(verse) and \
                has_enough_lines(verse) and \
                verse['lyrics'] not in verse_lyrics_set
            verse_lyrics_set.add(verse['lyrics'])
            if verse['valid']:
                verse['artist_id'] = artist_list.index(verse['artists'][0]) + 1
        for verse in song['verses']:
            if verse['valid']:
                return song
        # if all verses are invalid, remove the song
        return False
    return process_song

def mark_verses(song_list_path):
    song_list = read_list_from_file(song_list_path)
    artist_list = read_list_from_file('final_artist_list.txt')
    process_song = get_process_song(artist_list)
    loop_and_process(song_list, process_song, "Song", lambda x: x, marked_verses_dir)
    remove_duplicates_from_list_file(marked_verses_dir, skipped_artists_file)

if __name__ == "__main__":
    mark_verses('verses_with_artists/_LIST')