import os
from shutil import copyfile
import re

n = []
for root, dirs, files in os.walk("logs", topdown=True):
    for name in files:
        n.append(os.path.join(root,name))

regex = re.compile(r'.*08_1[7-9].*info.txt')
source = list(filter(regex.match, n))

for file in source:
    dest = file.replace("logs/","")
    dest = os.path.join("final_data", dest.replace("/","_"))
    copyfile(file, dest)
