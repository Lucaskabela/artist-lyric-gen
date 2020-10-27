import json
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file, clean_artist_names
from mark_verses import marked_verses_dir
from verse_songs_extract_artists import verse_artists_dir
from tqdm import tqdm

def update_analysis(analysis, lyric_type, num, title):
    if num not in analysis[lyric_type]:
        analysis[lyric_type][num] = {'count': 1, 'songs' : {title}}
    else:
        analysis[lyric_type][num]['count'] = analysis[lyric_type][num]['count'] + 1
        analysis[lyric_type][num]['songs'].add(title)

def update_artist(artists, artist_name, num_songs, num_verses, num_lines):
    # update count of num songs, num verses, num lines for an artist
    if artist_name not in artists:
        artists[artist_name] = [num_songs, num_verses, num_lines]
    else:
        artists[artist_name][0] = artists[artist_name][0] + num_songs
        artists[artist_name][1] = artists[artist_name][1] + num_verses
        artists[artist_name][2] = artists[artist_name][2] + num_lines

def get_update_songs():
    artists_list = read_list_from_file("cleaned_artist_names.txt")
    def update_songs(songs, song_title, artist_name, num_lines):
        # increment the count of a verse for an artist for a song
        artist_index = artists_list.index(artist_name)
        # num verses of each artist for a given song
        if song_title not in songs:
            songs[song_title] = [0] * len(artists_list)
            songs[song_title][artist_index] = num_lines
        else:
            songs[song_title][artist_index] = songs[song_title][artist_index] + num_lines
    return update_songs

def analyze_verses(song_list_path, song_dir):
    song_list = read_list_from_file(song_list_path)
    analysis = {'verses': {}, 'lines': {}, 'words': {}, 'words_per_verse': {}}
    artists = {}
    songs = {}
    bar = tqdm(song_list)
    update_songs = get_update_songs()
    for song_name in bar:
        # bar.set_description("Starting {}".format(song_name))
        with open("{}/{}.json".format(song_dir, name_to_file_name(song_name))) as song_file:
            song = json.load(song_file)
        title = song['title']
        num_verses = len(song['verses'])
        update_analysis(analysis, 'verses', num_verses, title)
        update_artist(artists, clean_artist_names(song['artist']).strip(), 1, 0, 0)
        # handle verses
        for verse in song['verses']:
            if not verse['valid']:
                continue
            # handle lines
            lines = [line.strip() for line in verse['lyrics'].split('\n')]
            lines = list(filter(lambda s: s != '', lines))
            num_lines = len(lines)
            update_analysis(analysis, 'lines', num_lines, title)
            update_artist(artists, verse['artists'][0], 0, 1, num_lines)
            update_songs(songs, "{} || {}".format(song['artist'], song['title']), verse['artists'][0], num_lines)
            # handle words
            total_words = 0
            for line in lines:
                words = [word.strip() for word in line.split()]
                words = list(filter(lambda s: s != '', words))
                num_words = len(words)
                total_words = total_words + num_words
                update_analysis(analysis, 'words', num_words, title)
            update_analysis(analysis, 'words_per_verse', total_words, title)
    for key in analysis.keys():
        for num in analysis[key].keys():
            analysis[key][num]['songs'] = list(analysis[key][num]['songs'])
    with open("verse_analysis.json", "w") as outfile:
        json.dump(analysis, outfile)
    with open("artist_analysis.json", "w") as outfile:
        json.dump(artists, outfile)
    with open("song_analysis.json", "w") as outfile:
        json.dump(songs, outfile)

if __name__ == "__main__":
    analyze_verses('marked_verses/_LIST', 'marked_verses')