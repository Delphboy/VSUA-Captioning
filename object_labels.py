import json

import numpy as np


def json_func():
    # read in the json file data/objects.json
    with open("data/objects.json", "r") as f:
        objects = json.load(f)

    dataset = {}

    for i, object in enumerate(objects):
        for j, obj in enumerate(object["objects"]):
            print(f"{i+1}-{j+1} / {len(objects)}-{len(object['objects'])}")

            if obj["object_id"] in dataset:
                assert (
                    dataset[obj["object_id"]] == obj["names"][0]
                ), f"Object {obj['object_id']} has multiple names: {dataset[obj['object_id']]} and {obj['names'][0]}"
            dataset[int(obj["object_id"])] = obj["names"][0]

    # sort the dataset dict by the key
    dataset = dict(sorted(dataset.items()))

    # write the dataset to data/object_labels.json
    with open("data/object_labels.json", "w") as f:
        json.dump(dataset, f)


def load_cocotalk():
    with open("data/cocotalk.json", "r") as f:
        cocotalk = json.load(f)
    x = cocotalk["ix_to_word"]
    return {int(k): v for k, v in x.items()}


def load_sg_dict():
    sg_dict = np.load("data/spice_sg_dict2.npz", allow_pickle=True)["spice_dict"][()]
    sg_dict = sg_dict["ix_to_word"]
    return sg_dict


if __name__ == "__main__":
    # Read in spice_sg_dict2.npz

    sg_dict = load_sg_dict()
    coco_talk = load_cocotalk()
    coco_inv = {v: k for k, v in coco_talk.items()}

    # Create a dictionary that maps the keys of spice_sg_dict2.npz to the keys of cocotalk.json where the values are the same
    # If the value is not the same, then print the key and the values
    mapping = {}
    for k, v in sg_dict.items():
        mapping[k] = coco_inv.get(v, 9487)

    ############################

    assert len(sg_dict) == len(
        mapping
    ), "The length of the two dictionaries are not the same"

    # assert all the keys in sg_dict are in mapping
    assert all(
        [x in mapping for x in sg_dict.keys()]
    ), "Not all the keys in sg_dict are in mapping"

    # assert all the keys in coco_talk are in mapping as keys
    assert all(
        [x in mapping for x in coco_talk.keys()]
    ), "Not all the keys in coco_talk are in mapping"

    ############################

    # save the mapping to data/sg_to_coco.npz
    np.savez("data/objectid_to_cocotalkid.npz", mapping=mapping)

    # load the mapping from data/sg_to_coco.npz
    sg_to_coco = np.load("data/objectid_to_cocotalkid.npz", allow_pickle=True)[
        "mapping"
    ][()]

    ############################

    assert len(sg_dict) == len(
        mapping
    ), "The length of the two dictionaries are not the same"

    # assert all the keys in sg_dict are in mapping
    assert all(
        [x in mapping for x in sg_dict.keys()]
    ), "Not all the keys in sg_dict are in mapping"

    # assert all the keys in coco_talk are in mapping as keys
    assert all(
        [x in mapping for x in coco_talk.keys()]
    ), "Not all the keys in coco_talk are in mapping"
