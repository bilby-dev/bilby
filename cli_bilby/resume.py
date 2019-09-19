""" A command line interface for converting resume files into results files """
import argparse
import os
import pickle

import pandas as pd
import bilby as bilby


def setup_command_line_args():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        "resume_files", nargs='+', help="List of resume files")
    parser.add_argument(
        "-f", '--format', default="json", help="Output format, defaults to json",
        choices=["json", "hdf5", "dat"])
    args = parser.parse_args()
    return args


def check_file(resume_file):
    """ Verify the file exists and is a resume file """
    if "resume.pickle" not in resume_file:
        raise ValueError("File {} is not a resume file".format(resume_file))
    if os.path.isfile(resume_file) is False:
        raise ValueError("No file {}".format(resume_file))


def get_outdir_and_label(resume_file):
    """ Infer the appropriate outdir and label from the resume file name """
    label = os.path.basename(resume_file).replace("_resume.pickle", "")
    outdir = os.path.dirname(resume_file)
    return outdir, label


def read_in_pickle_file(resume_file):
    """ Read in the pickle file

    Parameters
    ----------
    resume_file: str
        Input resume file path

    Returns
    -------
    df: pandas.DataFrame
        A data frame of the posterior

    """
    with open(resume_file, "rb") as file:
        data = pickle.load(file)

    if "posterior" in data:
        posterior = data["posterior"]
    else:
        raise ValueError("Resume file has no posterior, unable to convert")

    if "search_parameter_keys" in data:
        search_parameter_keys = data["search_parameter_keys"]
    else:
        search_parameter_keys = ["x{}".format(i) for i in range(posterior.shape[1])]

    df = pd.DataFrame(posterior, columns=search_parameter_keys)
    return df


def convert_df_to_posterior_samples(df, resume_file):
    filename = resume_file.replace("pickle", "dat")
    filename = filename.replace("resume", "preresult")
    df.to_csv(filename, index=False, header=True, sep=' ')


def convert_df_to_preresult(df, format, resume_file):
    outdir, label = get_outdir_and_label(resume_file)
    result = bilby.core.result.Result(
        label=label, outdir=outdir, search_parameter_keys=list(df.keys()))
    result.posterior = df
    result.priors = dict()
    filename = bilby.core.result.result_file_name(outdir, label, format)
    filename = filename.replace("result.{}".format(format), "preresult.{}".format(format))
    result.save_to_file(filename=filename, extension=format)


def convert_resume(resume_file, args):
    check_file(resume_file)
    print("Converting file {} to {}".format(resume_file, args.format))
    df = read_in_pickle_file(resume_file)
    if args.format == "dat":
        convert_df_to_posterior_samples(df, resume_file)
    elif args.format in ["json", "hdf5"]:
        convert_df_to_preresult(df, args.format, resume_file)


def main():
    args = setup_command_line_args()
    for resume_file in args.resume_files:
        convert_resume(resume_file, args)
