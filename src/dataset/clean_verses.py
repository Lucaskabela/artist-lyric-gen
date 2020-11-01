import json
import re
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file, clean_artist_names
from tqdm import tqdm
from raw_song_to_verse_split_song_files import verse_split_songs_dir

cleaned_verses_dir = "cleaned_verses_no_artists"
removed_verse_metadata_file = '_REMOVED_VERSES'

def clean_lyrics(s):
    # Do the cleaning steps
    cleaned_lyrics = s
    # Removes (...), {...}, * ... *
    cleaned_lyrics = re.sub(r'\(([^\)]+)\)|\*([^\*]+)\*|\{([^\}]+)\}', '', cleaned_lyrics)
    cleaned_lyrics = clean_artist_names(cleaned_lyrics)
    # Removes extra lines, and strips lines
    lines = [line.strip() for line in cleaned_lyrics.split('\n')]
    lines = list(filter(lambda s: s != '', lines))
    # reconstruct the lines together
    cleaned_lyrics = ''
    for line in lines:
        # concat lines together and add the end line token back
        cleaned_lyrics = cleaned_lyrics + line + '\n'
    # last line in a verse needs extra line break
    return cleaned_lyrics + '\n'

def process_song(song, bar):
    with open("{}/{}.json".format(verse_split_songs_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    new_verses = []
    for verse in song['verses']:
        new_lyrics = clean_lyrics(verse['lyrics'])
        verse['lyrics'] = new_lyrics
        if new_lyrics.strip() != '':
            # this is still a good verse, add it back
            # we remove the verses that are empty
            new_verses.append(verse)
        else:
            with open("{}/{}".format(cleaned_verses_dir, removed_verse_metadata_file), 'a') as openfile:
                openfile.write('{} || {}\n'.format(song['title'], verse['metadata']))
    song['verses'] = new_verses
    return song


def clean_verses(song_list_path, out_dir):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", lambda x:x, out_dir)