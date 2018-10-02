import argparse


def setup_command_line_args():
    parser = argparse.ArgumentParser(
        description="Plot corner plots from results files")
    parser.add_argument("-r", "--results", nargs='+',
                        help="List of results files to use.")
    parser.add_argument("-f", "--filename", default=None,
                        help="Output file name.")
    parser.add_argument("-l", "--labels", nargs='+', default=None,
                        help="List of labels to use for each result.")
    parser.add_argument("-p", "--parameters", nargs='+', default=None,
                        help="List of parameters.")
    parser.add_argument("-e", "--evidences", action='store_true', default=False,
                        help="Add the evidences to the legend.")
    args, _ = parser.parse_known_args()

    return args


def main():
    args = setup_command_line_args()
    import bilby
    results = [bilby.core.result.read_in_result(filename=r)
               for r in args.results]
    bilby.core.result.plot_multiple(results, filename=args.filename,
                                    labels=args.labels,
                                    parameters=args.parameters,
                                    evidences=args.evidences)
