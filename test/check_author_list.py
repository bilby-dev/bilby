""" A script to verify that the .AUTHOR.md file is up to date """

import re
import subprocess

special_cases = ["plasky", "thomas", "mj-will"]
AUTHORS_list = []
with open("AUTHORS.md", "r") as f:
    AUTHORS_list = " ".join([line for line in f]).lower()


lines = subprocess.check_output(["git", "shortlog", "HEAD", "-sn"]).decode("utf-8").split("\n")

if len(lines) == 0:
    raise Exception("No authors to check against")

fail_test = False
for line in lines:
    line = line.replace(".", " ")
    line = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', line))
    for element in line.split()[1:]:
        element = element.lower()
        if element not in AUTHORS_list and element not in special_cases:
            print(f"Failure: {element} not in AUTHOR.md")
            fail_test += True

if fail_test:
    raise Exception("Author check list failed.. have you added your name to the .AUTHOR file?")
