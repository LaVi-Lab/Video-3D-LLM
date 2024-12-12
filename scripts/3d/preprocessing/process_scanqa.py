import os
import json
import argparse

'''
{'answers': ['brown cabinet with tv sitting in it'], 
'object_ids': [8], 
'object_names': ['cabinet'], 
'question': 'What is in the right corner of room by curtains?', 
'question_id': 'train-scene0000-0', 
'scene_id': 'scene0000_00'}

{"id": 220602000037, 
"video": "scannet/scene0249_00", 
"conversations": [{"value": "<image> I am wiping a table that has a trash can next to it, while having a large table behind me with lots of chairs. Which direction should I toss a used napkin? Answer the question using a single word or phrase.", "from": "human"}, 
{"value": "right", "from": "gpt"}], 
"metadata": {"dataset": "sqa3d", "question_type": "which"}},
'''

def main(args):
    for split in ["train", "val"]:
        with open(os.path.join(args.scanqa_dir, f"ScanQA_v1.0_{split}.json")) as f:
            in_data = json.load(f)

        out_data = []

        for data in in_data:
            for idx, answer in enumerate(data['answers']):
                out = {}
                out['id'] = 'scanqa_{}_{}'.format(data['question_id'], idx)
                out['video'] = "scannet/{}".format(data['scene_id'])
                out['conversations'] = [
                    {
                        "value": "<image> {} Answer the question simply.".format(data['question']), 
                        "from": "human"
                    },
                    {
                        "value": answer,
                        "from": "gpt"
                    },
                ]
                out['metadata'] = {
                    "dataset": "scanQA", 
                    "question_type": "unknow",
                }
                if split == "val":
                    out["metadata"]["answers"] = data["answers"]
                    out_data.append(out)
                    break
                out_data.append(out)

        print(f"{len(out_data)} items...")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"scanqa_{split}_llava_style.json"), "w") as json_file:
            json.dump(out_data, json_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scanqa_dir", type=str, default="data/benchmark/scanqa")
    parser.add_argument("--output_dir", type=str, default="data/processed")

    args = parser.parse_args()
    main(args)

