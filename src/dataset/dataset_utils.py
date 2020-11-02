import json
import datetime
import time
import re
import traceback
import unidecode
import codecs
from subword_nmt import apply_bpe
from tqdm import tqdm

def name_to_file_name(name):
    return re.sub(r'\W+', '_', name)

def loop_and_process(
    objects, process_fn, item_type_name, get_obj_name_fn, data_dir, data_file_name_prefix=''
):
    i = 1
    start = time.time()
    os = tqdm(objects)
    for o in os:
        obj_name = get_obj_name_fn(o)
        obj_file_name = name_to_file_name(obj_name)
        try:
            # # os.set_description(
            #     "Starting {} {} out of {}: {}".format(
            #         item_type_name, i, len(objects), obj_name
            #     )
            # )
            processed_obj = process_fn(o, os)
            # os.set_description("Finished Processing {}".format(obj_name))
            # None means skip writing data to its own file, false means do not add to list
            if processed_obj is not None and processed_obj is not False:
                # write to own file
                with open(
                    "{}/{}{}.json".format(data_dir, data_file_name_prefix, obj_file_name), "w"
                ) as outfile:
                    json.dump(processed_obj, outfile)
                    # os.set_description("Wrote out data for {} to {}".format(obj_name, outfile.name))
            if processed_obj is not False:
                # add to the list
                with open("{}/_{}LIST".format(data_dir,data_file_name_prefix), "a") as outfile:
                    outfile.write("{}\n".format(obj_name))
                    # os.set_description("Wrote out {} to the list {}".format(obj_name, outfile.name))
            else:
                # removed from list
                with open("{}/_{}REMOVED".format(data_dir,data_file_name_prefix), "a") as outfile:
                    outfile.write("{}\n".format(obj_name))
                    # os.set_description("Wrote out {} to the removed list {}".format(obj_name, outfile.name))
            # os.set_description("Finished {}".format(obj_name))
        except Exception as e:
            tqdm.write("Failed to process {}".format(obj_name))
            with open("{}/_{}FAILED".format(data_dir,data_file_name_prefix), "a") as outfile:
                outfile.write("{}\n".format(obj_name))
            tqdm.write(e)
            traceback.print_exc()
        i = i + 1
    time_taken = str(datetime.timedelta(seconds=time.time() - start))
    print("{} for {} {}".format(time_taken, len(objects), item_type_name))

def read_list_from_file(list_path):
    with open(list_path, encoding="utf-8") as listfile:
        l = listfile.readlines()
    return [i.strip() for i in l]

def write_list_to_file(l, list_path, mode):
    with open(list_path, mode) as outfile:
        for i in l:
            outfile.write("{}\n".format(i))

def remove_duplicates_from_list_file(dir_name, file_name):
    print('Removing duplicates from list {}/{}'.format(dir_name, file_name))
    l = read_list_from_file('{}/{}'.format(dir_name, file_name))
    s = list(set(l))
    s.sort()
    write_list_to_file(s, '{}/{}'.format(dir_name, file_name), 'w')
    print('Done removing duplicates')

def compress_spaces(s):
    s = re.sub(r' +', ' ', s)
    return s

def remove_special_characters(s):
    # letter changing
    s = unidecode.unidecode(s)
    s = s.lower()
    # dollar signs for numbers
    # removes commas from numbers
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # changes $100 to 100 dollar
    s = re.sub(r"(\$)([\d,\.]+)", r"\2 dollar ", s)
    # changes non dollar sign $ to s
    s = re.sub(r"\$", "s", s)
    # change % to percent
    s = re.sub("\%", " percent ", s)
    # @ to at
    s = re.sub(r"\@\'n", 'at\'n', s)
    s = re.sub(r"\@", ' at ', s)
    # #1 to number 1
    s = re.sub(r"(\#)(\d)", r" number \2", s)
    # these symbols get removed
    s = re.sub(r"\'|\′|\u2019|\u00b4|\u2032|\`|\u2018", "", s)
    # everything else to spaces
    s = re.sub(r"[^a-zA-Z0-9\n]", " ", s)
    s = compress_spaces(s)
    return s

def clean_artist_names(s):
    #TODO: get the way yo clean artist names
    s = s.lower()
    s = re.sub(r"six\-nine|sixx\-nine", "6ix9ine", s)
    s = re.sub(r"^b\.ob\.", "b o b ", s)
    s = re.sub(r"^daddy kane", "big daddy kane", s)
    s = re.sub(r"busta rhyes|busta rymes|^busta$", "busta rhymes", s)
    s = re.sub(r"chamilionaire|chamillionare", "chamillionaire", s)
    s = re.sub(r"cheif keef", "chief keef", s)
    s = re.sub(r"p diddy|p\. diddy|p\.diddy", "diddy", s)
    s = re.sub(r"earl$", "earl sweatshirt", s)
    s = re.sub(r"elzhi\'s \"ego\"", "elzhi ", s)
    s = re.sub(r"fabulous|^fab$", "fabolous", s)
    s = re.sub(r"french montanna|frenchmontana|^french$", "french montana", s)
    s = re.sub(r"ghostace killah|ghostface$", "ghostface killah", s)
    s = re.sub(r"inspector deck|inspektah deck", "inspectah deck", s)
    s = re.sub(r"young jeezy", "jeezy", s)
    s = re.sub(r"^kanye$|k\. west|kayne west", "kanye west", s)
    s = re.sub(r"kendirck lamar|^kendrick$", "kendrick lamar", s)
    s = re.sub(r"^kodak$", "kodak black", s)
    s = re.sub(r"^krs$", "krs one", s)
    s = re.sub(r"^g rap$", "kool g rap", s)
    s = re.sub(r"lil wanye", "lil wayne", s)
    s = re.sub(r"^ll$|l\.l\. cool j", "ll cool j", s)
    s = re.sub(r"^luda$|ludracris", "ludacris", s)
    s = re.sub(r"^lupe$", "lupe fiasco", s)
    s = re.sub(r"masta ase|master ace", "masta ace", s)
    s = re.sub(r"^meth$|method$", "method man", s)
    s = re.sub(r"^doom$", "mf doom", s)
    s = re.sub(r"missy$|missy eliott|missy elliot$", "missy elliott", s)
    s = re.sub(r"nick minaj|^nicki$", "nicki minaj", s)
    s = re.sub(r"^obie$", "obie trice", s)
    s = re.sub(r"^r\.a$|ra the rugged man", "r a the rugged man", s)
    s = re.sub(r"^royce$", "royce da 59", s)
    s = re.sub(r"royce das", "royce da", s)
    s = re.sub(r"^snoop$|snopp dogg|snoop lion|snoop doggy dogg", "snoop dogg", s)
    s = re.sub(r"^t\.ii\.|^ti$", "t i ", s)
    s = re.sub(r"talib$", "talib kweli", s)
    s = re.sub(r"^tech$", "tech n9ne", s)
    s = re.sub(r"the notorioui b\.i\.g\.|the notorioui b\.i\.g\.|^notorious b\.i\.g|^notorious b\.i\.g\.|notorious big", "the notorious b i g ", s)
    s = re.sub(r"^vinnie$", "vinnie paz", s)
    s = re.sub(r"wiz khalfia", "wiz khalifa", s)
    s = remove_special_characters(s)
    return s

def get_bpe_object(codes_file_path):
    codes = codecs.open(codes_file_path, encoding='utf-8')
    bpe = apply_bpe.BPE(codes)
    codes.close()
    return bpe

def apply_bpe_to_string(s, bpe=None, codes_file_path=None):
    assert bpe is not None or codes_file_path is not None
    if bpe is None:
        bpe = get_bpe_object(codes_file_path)
    return bpe.process_line(s)

def revert_bpe(s):
    return re.sub(r'(@@ )|(@@ ?$)', '', s)
