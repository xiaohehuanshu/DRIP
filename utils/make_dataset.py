import json
import os
import random
import pandas as pd


def single_floor_prompt(input_condition, input_names, house_type, floor_num, image_path):
    # Build a single ShareGPT message
    sharegpt_entry = {
        "messages": [
            {
                "content": (
                    f"<image>\n请根据这张图片中已有的户型信息以及对应的参数，帮我生成其余房间的参数，得到一个完整的合理平面布局。"
                    f"构成户型的所有空间单元均表示为矩形，用x轴坐标、y轴坐标、长度、宽度四个参数表示。"
                    f"本户型为{house_type}住宅，图片中的为{floor_num}平面。\n"
                    f"图片中已有信息对应的参数为：\n"
                    f"```json\n{input_condition}\n```"
                    f"其余待生成的{floor_num}房间的名称为：\n```json\n{input_names}```"
                ),
                "role": "user"
            }
        ],
        "images": [image_path]
    }
    return sharegpt_entry


def second_floor_prompt(input_condition, input_names, house_type, floor_num, down_stair_condition, image_path):
    # Build a single ShareGPT message
    sharegpt_entry = {
        "messages": [
            {
                "content": (
                    f"<image>\n请根据这张图片中已有的户型信息以及对应的参数，帮我生成其余房间的参数，得到一个完整的合理平面布局。"
                    f"构成户型的所有空间单元均表示为矩形，用x轴坐标、y轴坐标、长度、宽度四个参数表示。"
                    f"本户型为{house_type}住宅，图片中的为{floor_num}平面。\n"
                    f"图片中已有信息对应的参数为：\n"
                    f"```json\n{input_condition}\n```"
                    f"首层平面的参数为：{down_stair_condition}\n"
                    f"其余待生成的{floor_num}房间的名称为：\n```json\n{input_names}```"
                ),
                "role": "user"
            }
        ],
        "images": [image_path]
    }
    return sharegpt_entry


def generate_prompt_from_file(df_in_data, room_lis, file_name, image_file_path, original_file_path=None):
    """
    Generate ShareGPT message from a given file.
    :param df_in_data: Input data DataFrame.
    :param room_lis: List of room names.
    :param file_name: File name without extension.
    :param image_file_path: Image file path.
    :param original_file_path: Original data file path (for second floor layouts).
    :return: ShareGPT message
    """
    # Convert to dictionary format
    dic_in_data = {col: df_in_data[col].tolist() for col in df_in_data.columns}

    # Determine house type and floor based on filename
    lis = file_name.split('_')

    house_type, floor_num = None, None
    if ('rural' in lis) and ('f' in lis):
        house_type = random.choice(['乡村', '自建'])
        floor_num = '一层'
    elif ('rural' in lis) and ('s' in lis):
        house_type = random.choice(['乡村', '自建'])
        floor_num = '二层'
    elif 'city' in lis:
        house_type = '城市'
        floor_num = '一层'

    # Build ShareGPT message
    if floor_num == '二层' and original_file_path:
        df_first_floor = pd.read_csv(original_file_path, index_col=0, encoding='utf_8_sig')
        dic_first_floor = {col: df_first_floor[col].tolist() for col in df_first_floor.columns}

        sharegpt_entry = second_floor_prompt(
            input_condition=json.dumps(dic_in_data, ensure_ascii=False),
            input_names=json.dumps(room_lis, ensure_ascii=False),
            house_type=house_type,
            floor_num=floor_num,
            down_stair_condition=json.dumps(dic_first_floor, ensure_ascii=False),
            image_path=image_file_path
        )
    else:
        sharegpt_entry = single_floor_prompt(
            input_condition=json.dumps(dic_in_data, ensure_ascii=False),
            input_names=json.dumps(room_lis, ensure_ascii=False),
            house_type=house_type,
            floor_num=floor_num,
            image_path=image_file_path
        )

    return sharegpt_entry


def process_and_save_prompts(input_data_path, output_data_path, image_data_path, original_data_path, output_json_path):
    """
    Batch process files and save as ShareGPT format JSON.
    :param input_data_path: Input data folder path.
    :param output_data_path: Output data folder path.
    :param image_data_path: Image folder path.
    :param original_data_path: Original data folder path (for second floor layouts).
    :param output_json_path: Final JSON file path to save.
    """
    sharegpt_data = []

    for root, dirs, files in os.walk(input_data_path):
        for idx, file in enumerate(files):
            print(f"Processing file {idx + 1}/{len(files)}: {file}")

            input_file = os.path.join(input_data_path, file)
            output_file = os.path.join(output_data_path, file)
            image_file = os.path.join(image_data_path, f"{os.path.splitext(file)[0]}.jpeg")

            # Determine if it's a second floor layout
            original_file = None
            if 'rural' in file and 's' in file:
                if 'mix' in file:
                    original_file = os.path.join(original_data_path, f"{file[:-8]}.csv")
                else:
                    original_file = os.path.join(original_data_path, f"{file[:-4]}.csv")

            # Call generation method
            sharegpt_entry = generate_prompt_from_file(input_file, output_file, image_file, original_file)
            sharegpt_data.append(sharegpt_entry)

    # Save as JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(sharegpt_data, json_file, ensure_ascii=False, indent=2)

    print(f"All prompts have been saved to {output_json_path}")


if __name__ == "__main__":
    process_and_save_prompts(
        input_data_path='data_merge_amplify/input_data/',
        output_data_path='data_merge_amplify/output_data/',
        image_data_path='data_merge_amplify/input_image/',
        original_data_path='data_merge/',
        output_json_path='dataset_house_floor.json'
    )
