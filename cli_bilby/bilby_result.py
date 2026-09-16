""" A command line interface to ease the process of batch jobs on result files

Examples
--------
To convert all the JSON result files in `outdir` to hdf5 format:

    $ bilby_result -r outdir/*json -e hdf5

Note, by default this will save the new files in the outdir defined within the
results files. To give a new location, use the `--outdir` argument.

To print the version and `log_evidence` for all the result files in outdir:

    $ bilby_result -r outdir/*json --print version log_evidence

To generate a corner plot for all the files in outdir

    $ bilby_result -r outdir/*json --call plot_corner

This is calling `plot_corner()` on each of the result files
individually. Note that passing extra commands in is not yet implemented.

"""
import argparse

import bilby
from bilby.core.result import EXTENSIONS
from bilby.core.utils import tcolors


def setup_command_line_args():
    parser = argparse.ArgumentParser(description="Helper tool for bilby result files")
    parser.add_argument("positional_results", nargs="*", help="List of results files.")
    parser.add_argument(
        "-r",
        "--results",
        nargs="+",
        dest="optional_results",
        default=list(),
        help="List of results files (alternative to passing results as a positional argument).",
    )

    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        choices=EXTENSIONS,
        default=True,
        help="Use given extension for the output file.",
    )
    parser.add_argument(
        "-g",
        "--gzip",
        action="store_true",
        help="Gzip the merged output results file if using JSON format.",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=None, help="Output directory."
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default=None,
        help="New label for output result object",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples for output result object",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="If true, strip back the result to only the posterior",
    )
    parser.add_argument(
        "--ignore-inconsistent",
        action="store_true",
        help="If true, ignore inconsistency errors in the merge process, but print a warning",
    )

    action_parser = parser.add_mutually_exclusive_group(required=True)
    action_parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save each result output saved using the outdir and label",
    )
    action_parser.add_argument(
        "-m",
        "--merge",
        action="store_true",
        help="Merge the set of runs, output saved using the outdir and label",
    )

    action_parser.add_argument(
        "-b", "--bayes", action="store_true", help="Print all Bayes factors."
    )
    action_parser.add_argument(
        "-p",
        "--print",
        nargs="+",
        default=None,
        dest="keys",
        help="Result dictionary keys to print.",
    )
    action_parser.add_argument(
        "--call",
        nargs="+",
        default=None,
        help="Result dictionary methods to call (no argument passing available).",
    )
    action_parser.add_argument(
        "--ipython",
        action="store_true",
        help=(
            "For each result given, drops the user into an "
            "IPython shell with the result loaded in"
        ),
    )
    args = parser.parse_args()

    args.results = args.positional_results + args.optional_results

    if len(args.results) == 0:
        raise ValueError("You have not passed any results to bilby_result")

    return args


def print_bayes_factors(results_list):
    for res in results_list:
        print(f"For result {res.label}:")
        print(f"  log_evidence={res.log_evidence}")
        print(f"  log_noise_evidence={res.log_noise_evidence}")
        print(f"  log_bayes_factor={res.log_noise_evidence}")
        print(f"  log_10_evidence={res.log_10_evidence}")
        print(f"  log_10_noise_evidence={res.log_10_noise_evidence}")
        print(f"  log_10_bayes_factor={res.log_noise_evidence}")


def drop_to_ipython(results_list):
    for result in results_list:
        message = "Opened IPython terminal for result {}".format(result.label)
        message += "\nRunning with bilby={},\nResult generated with bilby={}".format(
            bilby.__version__, result.version
        )
        message += "\nBilby result loaded as `result`"
        import IPython

        IPython.embed(header=message)


def print_matches(results_list, args):
    for r in results_list:
        print("\nResult file: {}/{}".format(r.outdir, r.label))
        for key in args.keys:
            for attr in r.__dict__:
                if key in attr:
                    print_line = [
                        "  ",
                        tcolors.KEY,
                        attr,
                        ":",
                        tcolors.VALUE,
                        str(getattr(r, attr)),
                        tcolors.END,
                    ]
                    print(" ".join(print_line))


def apply_max_samples(result, args):
    if len(result.posterior) > args.max_samples:
        result.posterior = result.posterior.sample(args.max_samples).sort_index()
    return result


def apply_lightweight(result, args):
    strip_keys = [
        "_nested_samples",
        "log_likelihood_evaluations",
        "log_prior_evaluations"
    ]
    for key in strip_keys:
        setattr(result, key, None)
    return result


def save(result, args):
    if args.max_samples is not None:
        result = apply_max_samples(result, args)
    if args.lightweight:
        result = apply_lightweight(result, args)
    result.save_to_file(gzip=args.gzip, extension=args.extension, outdir=args.outdir)


def main():
    args = setup_command_line_args()
    results_list = bilby.core.result.read_in_result_list(args.results)

    if args.save:
        for result in results_list:
            if args.label is not None:
                result.label = args.label
            save(result, args)

    if args.merge:
        if args.ignore_inconsistent:
            consistency_level = "warning"
        else:
            consistency_level = "error"
        result = results_list.combine(consistency_level=consistency_level)
        if args.label is not None:
            result.label = args.label
        if args.outdir is not None:
            result.outdir = args.outdir

        save(result, args)

    if args.keys is not None:
        print_matches(results_list, args)
    if args.call is not None:
        for r in results_list:
            for call in args.call:
                getattr(r, call)()
    if args.bayes:
        print_bayes_factors(results_list)
    if args.ipython:
        drop_to_ipython(results_list)
