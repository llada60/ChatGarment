import json
import os

final_dir = "/home/ids/liliu/data/ChatGarment/training/synthetic/images"
based_dir = "/home/ids/liliu/data/ChatGarment/training/synthetic/single_sketch"

for file in os.listdir(based_dir):
    if file.endswith(".json"):
        json_path = os.path.join(based_dir, file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'detailtext' not in file:
            garment_dict = {}
            for item in data:
                garment_id = item['garment_id']
                if garment_id not in garment_dict:
                    garment_dict[garment_id] = item.copy()
                    garment_dict[garment_id]['sketch_path'] = []
                garment_dict[garment_id]['sketch_path'].append(item['sketch_path'].replace('/sketch/','/img/'))
            data = []
            for garment_id, item in garment_dict.items():
                item['sketch_num'] = len(item['sketch_path'])
                data.append(item)
        else:
            for item in data:
                if 'sketch_num' in item:
                    del item['sketch_num']
                if 'sketch_path' in item:
                    del item['sketch_path']

        with open(os.path.join(final_dir, file), 'w') as f:
            json.dump(data, f)
            