import json
from dataset_utils import loop_and_process, name_to_file_name, read_list_from_file, write_list_to_file, remove_duplicates_from_list_file
from mark_verses import marked_verses_dir
from verse_songs_extract_artists import verse_artists_dir
from tqdm import tqdm

def update_analysis(analysis, lyric_type, num, title):
    if num not in analysis[lyric_type]:
        analysis[lyric_type][num] = {'count': 1, 'songs' : {title}}
    else:
        analysis[lyric_type][num]['count'] = analysis[lyric_type][num]['count'] + 1
        analysis[lyric_type][num]['songs'].add(title)

def analyze_verses(song_list_path, song_dir):
    song_list = read_list_from_file(song_list_path)
    analysis = {'verses': {}, 'lines': {}, 'words': {}, 'words_per_verse': {}}
    bar = tqdm(song_list)
    for song_name in bar:
        # bar.set_description("Starting {}".format(song_name))
        with open("{}/{}.json".format(song_dir, name_to_file_name(song_name))) as song_file:
            song = json.load(song_file)
        title = song['title']
        num_verses = len(song['verses'])
        update_analysis(analysis, 'verses', num_verses, title)
        # handle lines
        for verse in song['verses']:
            lines = [line.strip() for line in verse['lyrics'].split('\n')]
            lines = list(filter(lambda s: s != '', lines))
            num_lines = len(lines)
            update_analysis(analysis, 'lines', num_lines, title)
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
