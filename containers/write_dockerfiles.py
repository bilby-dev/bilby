from datetime import date

with open("dockerfile-template", "r") as ff:
    template = ff.read()

python_versions = [(3, 5), (3, 6), (3, 7), (3, 8)]
today = date.today().strftime("%Y%m%d")

for python_major_version, python_minor_version in python_versions:
    with open(
        "v2-dockerfile-test-suite-python"
        f"{python_major_version}{python_minor_version}",
        "w"
    ) as ff:
        ff.write(
            "# This dockerfile is written automatically and should not be "
            "modified by hand.\n\n"
        )
        ff.write(template.format(
            date=today,
            python_major_version=python_major_version,
            python_minor_version=python_minor_version
        ))
