import argparse
import lyricsgenius
import pandas as pd
import time
from dataset_utils import loop_and_process
from genius import GENIUS_ACCESS_TOKEN

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name of an artist")
parser.add_argument("--id", type=int, help="Id of the artist")
args = parser.parse_args()

if __name__ == "__main__":

    def process_artist(row):
        _, row = row
        artist = genius.search_artist(row["Artist"], artist_id=row["Id"])
        artist.save_lyrics()
        return {"artist": row["Artist"], "num_songs": artist.num_songs}

    def get_artist_name(row):
        _, row = row
        return row["Artist"]

    if args.name is None:
        print("\n Getting lyrics for all artists")
        artists = pd.read_csv("artists.csv")
        #   print('Artist {} out of {}'.format(index, len(artists)))
        #   start = time.time()
        #   artist = genius.search_artist(row["Artist"], artist_id=row["Id"])
        #   print("Num songs {}".format(len(artist['songs'])))
        #   artist.save_lyrics()
        #   print("Lyrics saved")
        #   print("--- %s seconds ---" % (time.time() - start))
    else:
        row = (None, {"Artist": args.name, "Id": args.id})
        loop_and_process(
            artists.iterrows(),
            process_artist,
            "Artist",
            get_artist_name,
            "Artists_Scraped",
        )
        # for index, row in artists.iterrows():
