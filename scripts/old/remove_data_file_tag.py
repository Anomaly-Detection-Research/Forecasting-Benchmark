import helpers
import os
import re

# args
csv_input_directory = "../data"

# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
cvs_files = []
for path, dnames, fnames in os.walk(csv_input_directory):
    cvs_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

print len(cvs_files)
for f in cvs_files:
    os.rename(f, helpers.remove_tag_from_file(f))