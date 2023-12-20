import argparse
import os
import sys


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


def print_results(results):
    print(results["Bleu_1"])
    print(results["Bleu_2"])
    print(results["Bleu_3"])
    print(results["Bleu_4"])
    print(results["METEOR"])
    print(results["ROUGE_L"])
    print(results["CIDEr"])


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


if __name__ == "__main__":
    # Get the file directory using argparse
    parser = argparse.ArgumentParser(description="Get the file directory")
    parser.add_argument("-f", type=str, default="log", help="file directory")
    args = parser.parse_args()

    # Get the file directory
    file_dir = args.f

    # Get the list of files in the directory
    files = os.listdir(file_dir)
    files = [file for file in files if file.__contains__(".o")]
    for file in files:
        lines = read_file(os.path.join(args.f, file))
        if check_for_error(lines):
            print(file)
            continue

        params = get_model_params_from_file_name(file)
        print(
            f"Seed 0 | Percentage {params['percentage']} | RNN Size {params['rnn_size']} | RNN Layers {params['rnn_layers']} | Input Encoding Size {params['input_encoding_size']} | Attn Hidden Size {params['attn_hidden_size']} | Grad Clip {params['grad_clip']} | Dropout {params['dropout']} | Graph Type {params['graph_type']} | Weighted {params['weighted']}"
        )
        print_results(read_file_bottom_up(lines))
        print()
