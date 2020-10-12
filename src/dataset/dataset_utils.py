import json
import datetime
import time
import re
from tqdm import tqdm

def name_to_file_name(name):
    return re.sub(r'\W+', '_', name)

def loop_and_process(
    objects, process_fn, item_type_name, get_obj_name_fn, data_dir, data_file_name_prefix=''
):
    i = 1
    print("{} total {}".format(len(objects), item_type_name))
    start = time.time()
    for o in tqdm(objects):
        obj_name = get_obj_name_fn(o)
        obj_file_name = name_to_file_name(obj_name)
        try:
            tqdm.write(
                "Starting {} {} out of {}: {}".format(
                    item_type_name, i, len(objects), obj_name
                )
            )
            processed_obj = process_fn(o)
            tqdm.write("Finished Processing {}".format(obj_name))
            if processed_obj is not None:
                with open(
                    "{}/{}{}.json".format(data_dir, data_file_name_prefix, obj_file_name), "w"
                ) as outfile:
                    json.dump(processed_obj, outfile)
                    tqdm.write("Wrote out data for {} to {}".format(obj_name, outfile.name))
            with open("{}/{}LIST".format(data_dir,data_file_name_prefix), "a") as outfile:
                outfile.write("{}\n".format(obj_name))
                tqdm.write("Wrote out {} to the list {}".format(obj_name, outfile.name))
            tqdm.write("Finished {}".format(obj_name))
        except Exception as e:
            print("Failed to process {}".format(obj_name))
            with open("{}/{}FAILED".format(data_dir,data_file_name_prefix), "a") as outfile:
                outfile.write("{}\n".format(obj_name))
            print(e)
        i = i + 1
    time_taken = str(datetime.timedelta(seconds=time.time() - start))
    print("{} for {} {}".format(time_taken, len(objects), item_type_name))
