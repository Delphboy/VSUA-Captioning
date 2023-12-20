import argparse
import os
import sys

# file_name is in the structure of:
# seed{seed}_split{split}_percentage_rnnSize_rnnLayers_inputEncodingSize_attnHiddenSize_gradClip_dropout_graphType_weightd.output
# example: seed1_split2_0.25_1000_1_1000_512_0.1_0.5_geometry_unweighted.output
# example: seed1_split2_0.25_1000_1_1000_512_0.1_0.5_geometry_weighted.output


def print_results(files: list, name: str) -> None:
    results = []
    for seed in seeds:
        for split in splits:
            file = [file for file in files if f"seed{seed}_split{split}" in file][0]

            with open(os.path.join(args.directory, file), "r") as f:
                lines = f.readlines()
                if check_for_error(lines):
                    print(f"Error detected in {file}")
                    continue
                results.append(read_file_bottom_up(lines))

    # Average the results for all the metrics
    average_results = {}
    for metric in results[0]:
        average_results[metric] = sum(
            [float(result[metric]) for result in results]
        ) / len(results)

    # Find the standard error of all the seeds
    standard_error_results = {}
    for metric in results[0]:
        standard_error_results[metric] = (
            sum(
                [
                    (float(result[metric]) - average_results[metric]) ** 2
                    for result in results
                ]
            )
            / len(results)
        ) ** 0.5

    print(name)
    print(
        f"{average_results['Bleu_1']:.3f}\t{average_results['Bleu_2']:.3f}\t{average_results['Bleu_3']:.3f}\t{average_results['Bleu_4']:.3f}\t{average_results['METEOR']:.3f}\t{average_results['ROUGE_L']:.3f}\t{average_results['CIDEr']:.3f}"
    )
    print(
        f"{standard_error_results['Bleu_1']:.3f}\t{standard_error_results['Bleu_2']:.3f}\t{standard_error_results['Bleu_3']:.3f}\t{standard_error_results['Bleu_4']:.3f}\t{standard_error_results['METEOR']:.3f}\t{standard_error_results['ROUGE_L']:.3f}\t{standard_error_results['CIDEr']:.3f}"
    )
    print("-" * CARRIDGE_WIDTH)


def print_args(args):
    print("Arguments:")
    for arg in vars(args):
        if arg != "directory":
            print(f"\t{arg}: {getattr(args, arg)}")


def check_for_error(lines):
    for line in reversed(lines):
        if "Traceback (most recent call last):" in line:
            return True


def read_file_bottom_up(lines):
    results = {}
    for line in reversed(lines):
        if "Bleu_1:" in line:
            results["Bleu_1"] = line.split(":")[1].strip()
            break
        if "Bleu_2:" in line:
            results["Bleu_2"] = line.split(":")[1].strip()
        if "Bleu_3:" in line:
            results["Bleu_3"] = line.split(":")[1].strip()
        if "Bleu_4:" in line:
            results["Bleu_4"] = line.split(":")[1].strip()
        if "METEOR:" in line:
            results["METEOR"] = line.split(":")[1].strip()
        if "ROUGE_L:" in line:
            results["ROUGE_L"] = line.split(":")[1].strip()
        if "CIDEr:" in line:
            results["CIDEr"] = line.split(":")[1].strip()
    return results


def build_file_name_body_from_hyperparams(args):
    return f"{args.percentage}_{args.rnnSize}_{args.rnnLayers}_{args.inputEncodingSize}_{args.attnHiddenSize}_{args.gradClip}_{args.dropout}"


def get_files_in_dir(dir):
    return os.listdir(dir)


def get_model_params_from_file_name(file_name) -> dict:
    splits = file_name.split("_")
    seed = splits[0][1:]
    split = splits[1][5:]
    percentage = splits[2]
    rnn_size = splits[3]
    rnn_layers = splits[4]
    input_encoding_size = splits[5]
    attn_hidden_size = splits[6]
    grad_clip = splits[7]
    dropout = splits[8]
    graph_type = splits[9]
    weighted = splits[10].split(".")[0]

    return {
        "seed": seed,
        "split": split,
        "percentage": percentage,
        "rnn_size": rnn_size,
        "rnn_layers": rnn_layers,
        "input_encoding_size": input_encoding_size,
        "attn_hidden_size": attn_hidden_size,
        "grad_clip": grad_clip,
        "dropout": dropout,
        "graph_type": graph_type,
        "weighted": weighted,
    }


def get_seed_from_file_name(file_name) -> str:
    return file_name[4]


def get_split_from_file_name(file_name) -> str:
    return file_name[11]


if __name__ == "__main__":
    print()
    CARRIDGE_WIDTH = 53
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="path to log files",
        default="log",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        required=False,
        help="percentage of training data",
        default=0.25,
    )
    parser.add_argument(
        "--rnnSize", type=int, required=False, help="rnn size", default=1000
    )
    parser.add_argument(
        "--rnnLayers", type=int, required=False, help="rnn layers", default=1
    )
    parser.add_argument(
        "--inputEncodingSize",
        type=int,
        required=False,
        help="input encoding size",
        default=1000,
    )
    parser.add_argument(
        "--attnHiddenSize",
        type=int,
        required=False,
        help="attention hidden size",
        default=1024,
    )
    parser.add_argument(
        "--gradClip", type=float, required=False, help="gradient clip", default=0.1
    )
    parser.add_argument(
        "--dropout", type=float, required=False, help="dropout", default=0.5
    )

    args = parser.parse_args()
    file_name_search = build_file_name_body_from_hyperparams(args)
    files = get_files_in_dir(args.directory)

    print("Analysing the results for the following hyperparameters:")
    print_args(args)
    print()

    # Get the list of files that match the hyperparameters
    files_to_analyse = [file for file in files if file_name_search in file]

    geometry_unweighted_files = [
        file for file in files_to_analyse if "geometry_unweighted" in file
    ]
    geometry_weighted_files = [
        file for file in files_to_analyse if "geometry_weighted" in file
    ]
    semantic_unweighted_files = [
        file for file in files_to_analyse if "semantic_unweighted" in file
    ]
    semantic_weighted_files = [
        file for file in files_to_analyse if "semantic_weighted" in file
    ]

    seeds = sorted(list(set([get_seed_from_file_name(f) for f in files_to_analyse])))
    splits = sorted(list(set([get_split_from_file_name(f) for f in files_to_analyse])))
    print("  B1\t  B2\t  B3\t  B4\t  M\t  R\t  C\t")
    print("=" * CARRIDGE_WIDTH)

    print_results(geometry_unweighted_files, "Geometry Unweighted")
    print_results(geometry_weighted_files, "Geometry Weighted")
    print_results(semantic_unweighted_files, "Semantic Unweighted")
    print_results(semantic_weighted_files, "Semantic Weighted")
