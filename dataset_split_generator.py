import argparse
import json
import os
import random
import sys

FILE_NAME = "data/dataset_coco.json"


def load_file() -> dict:
    with open(FILE_NAME, "r") as f:
        data = json.load(f)
    return data


def generate_json_input(percentage: float, seed: int):
    input_json_file_name = f"data/dataset_coco_{int(percentage * 100)}_{seed}.json"

    if os.path.exists(input_json_file_name):
        return

    print(f"Setting seed to {seed}")
    random.seed(seed)

    print(f"Generating {input_json_file_name}")
    data = load_file()
    train = []
    val = []
    test = []
    images = data["images"]
    for img in images:
        if img["split"] == "val":
            val.append(img)
        elif img["split"] == "test":
            test.append(img)
        else:
            train.append(img)

    # take a random sample of train of size percentage
    mini_train = random.sample(train, int(len(train) * percentage))
    mini_val = random.sample(val, int(len(val) * percentage))
    mini_test = random.sample(test, int(len(test) * percentage))

    new_data = {}
    new_data["images"] = mini_train + mini_val + mini_test
    new_data["dataset"] = data["dataset"]

    print(f"Number of images in train: {len(mini_train)}")
    print(f"Number of images in val: {len(mini_val)}")
    print(f"Number of images in test: {len(mini_test)}")
    print(f"Total number of images: {len(new_data['images'])}")

    # write new_data to json file
    with open(input_json_file_name, "w+") as f:
        json.dump(new_data, f)

    print(f"Generated {input_json_file_name}")


def do_output_files_exist(output_json_file_name: str, output_h5_file_name: str):
    output_json_file_name = os.path.join(os.getcwd(), output_json_file_name)
    output_h5_file_name = os.path.join(os.getcwd(), output_h5_file_name)
    return os.path.exists(output_json_file_name) and os.path.exists(
        f"{output_h5_file_name}_label.h5"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the percentage")
    parser.add_argument("-p", type=float, default=0.50, help="percentage")
    parser.add_argument("-s", type=int, default=0, help="seed")
    args = parser.parse_args()
    percentage = args.p
    seed = args.s

    if percentage < 0 or percentage > 1:
        print("Percentage must be between 0 and 1")
        sys.exit(1)

    generate_json_input(percentage, seed)

    input_json_file_name = f"data/dataset_coco_{int(percentage * 100)}_{seed}.json"
    output_json_file_name = f"data/cocotalk_{int(percentage * 100)}_{seed}.json"
    output_h5_file_name = f"data/cocotalk_{int(percentage * 100)}_{seed}"

    if not do_output_files_exist(output_json_file_name, output_h5_file_name):
        os.system(
            f"python scripts/prepro_labels.py --input_json {input_json_file_name} --output_json {output_json_file_name} --output_h5 {output_h5_file_name}"
        )
