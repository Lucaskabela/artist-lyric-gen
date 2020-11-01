import argparse
import json
from dataset_utils import loop_and_process, name_to_file_name
from clean_lyrics import cleaned_songs_dir

verse_split_songs_dir = "raw_verse_songs"

def raw_songs_to_verse_split_songs(song_list_path, dir_prefix):
    with open(song_list_path) as listfile:
        song_list = listfile.readlines()
    song_list = [song.strip() for song in song_list]
    def process_song(song, bar):
        with open("{}/{}.json".format(cleaned_songs_dir, name_to_file_name(song))) as song_file:
            song = json.load(song_file)
        verses = []
        lyrics = song['lyrics']
        i = 0
        verse_lyrics = ''
        verse_metadata = ''
        def write_verse(v_metadata, v_lyrics, verses):
            if len(v_metadata.strip()) > 0:
                verses.append({'metadata': v_metadata, 'lyrics': v_lyrics})
            return verses
        while i < len(lyrics):
            # Parse the songs into each verse
            if lyrics[i] == '[':
                # we reached a new verse
                # append the previous verse if there was one
                verses = write_verse(verse_metadata, verse_lyrics, verses)
                # reset for the new verse
                verse_lyrics = ''
                verse_metadata = ''
                # start processing the new verse
                while i < len(lyrics) and lyrics[i] != ']':
                    verse_metadata = verse_metadata + lyrics[i]
                    i = i + 1
                if i < len(lyrics):
                    verse_metadata = verse_metadata + lyrics[i]
            else:
                verse_lyrics = verse_lyrics + lyrics[i]
            i = i + 1
        verses = write_verse(verse_metadata, verse_lyrics, verses)
        return {
            'title': song['title'],
            'verses': verses,
            'artist': song['artist'],
            'featured_artists': song['featured_artists']
        }
    def get_song_name(song):
        return song
    loop_and_process(song_list, process_song, "Song", get_song_name, dir_prefix)

if __name__ == "__main__":
    raw_songs_to_verse_split_songs('raw_songs/LIST', verse_split_songs_dir)