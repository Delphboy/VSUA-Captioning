import argparse
import os


# Read the file into a list
def read_file(file_dir):
    with open(file_dir, "r") as f:
        lines = f.readlines()
    return lines


# Check that there wasn't an error in the file
def check_for_error(lines):
    for line in reversed(lines):
        if "Traceback (most recent call last):" in line:
            return True


# read the file line by line from the bottom up
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


def get_model_params_from_file_name(file_name) -> dict:
    # file_name is in the structure of:
    # s{seed}_percentage_rnnSize_rnnLayers_inputEncodingSize_attnHiddenSize_gradClip_dropout_graphType_weightd.output
    # example: s1_0.25_1000_1_1000_512_0.1_0.5_geometry_unweighted.output
    # example: s1_0.25_1000_1_1000_512_0.1_0.5_geometry_weighted.output

    # Get the seed
    splits = file_name.split("_")
    seed = splits[0][1:]
    percentage = splits[1]
    rnn_size = splits[2]
    rnn_layers = splits[3]
    input_encoding_size = splits[4]
    attn_hidden_size = splits[5]
    grad_clip = splits[6]
    dropout = splits[7]
    graph_type = splits[8]
    weighted = splits[9].split(".")[0]

    return {
        "seed": seed,
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


def print_results(results):
    print(f"Bleu 1: {100 * float(results['Bleu_1'])}")
    print(f"Bleu 2: {100 * float(results['Bleu_2'])}")
    print(f"Bleu 3: {100 * float(results['Bleu_3'])}")
    print(f"Bleu 4: {100 * float(results['Bleu_4'])}")
    print(f"Meteor: {100 * float(results['METEOR'])}")
    print(f"Rogue: {100 * float(results['ROUGE_L'])}")
    print(f"CIDEr: {100 * float(results['CIDEr'])}")


if __name__ == "__main__":
    # Create a parser object that will take in a list of file names
    parser = argparse.ArgumentParser(description="Get the file directory")
    parser.add_argument("-f", type=str, default="log", help="file directory")
    args = parser.parse_args()

    # Get the file directory
    file_dir = args.f

    # Get the list of files in the directory
    files = os.listdir(file_dir)
    files = [file for file in files if file.__contains__(".o")]

    results = {}

    # Loop through the files
    for file in files:
        # Get the file directory
        file_dir = os.path.join(args.f, file)

        # Read the file into a list
        lines = read_file(file_dir)

        # Check that there wasn't an error in the file
        if check_for_error(lines):
            print("Error in file: {}".format(file))
            continue

        # read the file line by line from the bottom up
        result = read_file_bottom_up(lines)

        results[file] = result

    # Search queries
    seeds = ["0", "1", "2"]
    percents = ["0.1"]  # , "0.25", "0.5", "0.75", "1.0"]
    rnn_sizes = ["512"]  # , "1000"]
    rnn_layers = ["1"]  # , "2"]
    input_encoding_sizes = ["512"]  # , "1000"]
    attn_hidden_sizes = ["512"]  # , "1024"]
    graph_types = ["semantic"]  # , "semantic"]
    weights = ["unweighted"]  # , "weighted"]
    queries = []

    for s in seeds:
        for p in percents:
            for r in rnn_sizes:
                for ies in input_encoding_sizes:
                    for ahs in attn_hidden_sizes:
                        for gl in graph_types:
                            for w in weights:
                                queries.append(
                                    {
                                        "seed": s,
                                        "percentage": p,
                                        "rnn_size": r,
                                        "rnn_layers": "1",
                                        "input_encoding_size": ies,
                                        "attn_hidden_size": ahs,
                                        "grad_clip": "0.1",
                                        "dropout": "0.5",
                                        "graph_type": gl,
                                        "weighted": w,
                                    }
                                )

    # Loop through the queries
    for query in queries:
        best_file = None
        best_result = None
        for file in results:
            # Get the model params from the file name
            model_params = get_model_params_from_file_name(file)

            # Check if the model params match the query
            if (
                model_params["percentage"] == query["percentage"]
                and model_params["seed"] == query["seed"]
                # and model_params["rnn_size"] == query["rnn_size"]
                # and model_params["rnn_layers"] == query["rnn_layers"]
                # and model_params["input_encoding_size"] == query["input_encoding_size"]
                # and model_params["attn_hidden_size"] == query["attn_hidden_size"]
                # and model_params["grad_clip"] == query["grad_clip"]
                # and model_params["dropout"] == query["dropout"]
                and model_params["graph_type"] == query["graph_type"]
                and model_params["weighted"] == query["weighted"]
            ):
                # print(model_params)
                # print_results(results.get(file))
                # print()

                # Check if this result is better than the current best result
                if best_result is None or float(results.get(file).get("CIDEr")) > float(
                    best_result.get("CIDEr")
                ):
                    best_result = results[file]
                    best_file = file

        # Print the best result for this query
        if best_file is not None:
            print()
            print(f"Best result for query: {query}")
            print(f"File: {best_file}")
            print_results(best_result)
            print()
