""" A script to verify that the .AUTHOR.md file is up to date """

import re
import subprocess

special_cases = ["plasky", "thomas", "mj-will", "richard", "douglas", "nixnyxnyx"]
AUTHORS_list = []
with open("AUTHORS.md", "r") as f:
    AUTHORS_list = " ".join([line for line in f]).lower()


lines = subprocess.check_output(["git", "shortlog", "HEAD", "-sn"]).decode("utf-8").split("\n")

if len(lines) == 0:
    raise Exception("No authors to check against")


def remove_accents(raw_text):

    raw_text = re.sub(u"[àáâãäå]", 'a', raw_text)
    raw_text = re.sub(u"[èéêë]", 'e', raw_text)
    raw_text = re.sub(u"[ìíîï]", 'i', raw_text)
    raw_text = re.sub(u"[òóôõö]", 'o', raw_text)
    raw_text = re.sub(u"[ùúûü]", 'u', raw_text)
    raw_text = re.sub(u"[ýÿ]", 'y', raw_text)
    raw_text = re.sub(u"[ß]", 'ss', raw_text)
    raw_text = re.sub(u"[ñ]", 'n', raw_text)

    return raw_text


fail_test = False
for line in lines:
    line = line.replace(".", " ")
    line = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', line))
    line = remove_accents(line)
    for element in line.split()[1:]:
        element = element.lower()
        if element not in AUTHORS_list and element not in special_cases:
            print(f"Failure: {element} not in AUTHOR.md")
            fail_test += True

if fail_test:
    raise Exception("Author check list failed.. have you added your name to the .AUTHOR file?")
