import json
import re
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, clean_artist_names, remove_duplicates_from_list_file
from clean_verses import cleaned_verses_dir

verse_artists_dir = "verses_with_artists"
artists_list_file = "_ARTISTS_LIST.txt"
raw_artists_list_file = "_RAW_ARTISTS_LIST.txt"
bad_metadata_file = "_BAD_METADATA.txt"

def get_artists_from_metadata(song_title, metadata, song_artist, featured_artists):
    def write_out_error(song_title, metadata):
        with open("{}/{}".format(verse_artists_dir,bad_metadata_file), 'a') as openfile:
            openfile.write('{} || {}\n'.format(song_title, metadata))
    s = metadata
    # remove the [] that wraps the raw metadata
    if s[0] == "[":
        s = s[1:]
    if s[-1] == "]":
        s = s[:-1]
    # metadata should have the for <verse_type> : <artist1> & <artist2>...
    artists = s.split(':')
    if len(artists) > 2:
        write_out_error(song_title, metadata)
    elif len(artists) < 2:
        # we dont want to write out things that were like [verse] or [verse 1]
        cleaned_metadata = re.sub(r'[0-9]+', '', artists[0])
        if ' ' in cleaned_metadata.strip():
            write_out_error(song_title, metadata)
        # no artists were given so we default to the main artist
        artists = [song_artist]
    else:
        # gets the artists after : eg. [verse: <artist>]
        artists = artists[1]
        artists = artists.replace(')','')
        # get an actual list of artists
        artists = re.split(r'&|\+|,|and|\(', artists)
        # filter out the empty string splits
        artists = [artist.strip() for artist in artists]
        artists = set(filter(lambda s: s != '', artists))
        # if no artist given, set to default artist
        if len(artists) == 0:
            artists = [song_artist]
    # write out list of uncleaned artist names
    write_list_to_file(artists, "{}/{}".format(verse_artists_dir, raw_artists_list_file), 'a')
    # clean the artists names
    artists = [clean_artist_names(artist).strip() for artist in artists]
    # filter out null names
    artists = set(filter(lambda s: s != '', artists))
    # redefault to song artist and write out new name
    if len(artists) == 0:
        artists = [clean_artist_names(song_artist).strip()]
    write_list_to_file(artists, "{}/{}".format(verse_artists_dir, artists_list_file), 'a')
    return list(artists)

def process_song(song, bar):
    with open("{}/{}.json".format(cleaned_verses_dir, name_to_file_name(song))) as song_file:
        song = json.load(song_file)
    for verse in song['verses']:
        artists = get_artists_from_metadata(song['title'], verse['metadata'], song['artist'], song['featured_artists'])
        verse['artists'] = artists
    return song

def get_song_name(song):
    return song

def verse_songs_extract_artists(song_list_path, dir_prefix):
    song_list = read_list_from_file(song_list_path)
    loop_and_process(song_list, process_song, "Song", get_song_name, dir_prefix)
    remove_duplicates_from_list_file(verse_artists_dir, artists_list_file)

