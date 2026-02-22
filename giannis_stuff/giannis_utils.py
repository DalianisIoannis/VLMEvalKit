"""
VLMEvalKit Thesis Utilities Module

This module contains utility functions for thesis research on multimodal workload evaluation.
It supports:
- Video frame sampling techniques (scene change, motion-based, sharpness)
- Dataset preparation and manipulation (TSV/JSONL conversion, subsetting)
- Evaluation metrics (accuracy, ROUGE, BLEU, CIDEr)
- Visualization (CDFs, plots, frame analysis)
- Path and configuration management

Main Components:
----------------

1. VIDEO FRAME SAMPLING:
   - apply_clever_sampling(): Main entry point for intelligent frame selection
   - sample_frames_by_scene_change_decord_version(): Scene boundary detection
   - sample_frames_by_motion_based(): Motion-based frame selection
   - sample_frames_by_sharpness(): Sharpness-based frame selection
   - calculate_sharpness_laplacian(): Blur detection using Laplacian variance

2. DATASET UTILITIES:
   - create_small_tsv(): Create dataset subsets with optional shuffling
   - tsv_to_jsonl(): Dataset format conversion
   - make_image_path_column(): Add image path columns for compressed datasets
   - create_question_ids(): Generate unique question identifiers

3. EVALUATION METRICS:
   - calculate_accuracy_2_dataframe_columns(): Accuracy computation
   - calculate_accuracy_on_MMBench_DEV_EN(): MMBench-specific evaluation
   - compressed_data_scorer(): ROUGE, BLEU, CIDEr metrics for COCO captioning
   - evaluate_llm_scores(): LLaVABench evaluation with local judge

4. VISUALIZATION:
   - plot_simple_cdf(): Cumulative distribution plots
   - coco_visualizer(): COCO caption results visualization
   - plot_frame_sizes(): Video frame dimension analysis
   - save_figure_as_pdf(): PDF plot export

5. PATH CONFIGURATION:
   - HOME_DATA_PATH: Root data directory
   - LMUData_PATH: Dataset storage location
   - CACHE_MODEL_DIR: HuggingFace cache directory
   - OUTPUTS_FOLDER: Evaluation results storage

Constants:
----------
PIXEL_REDUCTIONS: Compression levels for image experiments (0-90%)
SMALL_TSV_SAMPLES: Default dataset subset size (350 samples)
PLOTS_SAVE_PATH: Directory for saving generated plots

Usage Examples:
---------------

# Apply clever sampling to a video
frames = apply_clever_sampling(
    video_path="/path/to/video.mp4",
    max_frames=64,
    technique="scene_change",
    param=27
)

# Create a dataset subset
create_small_tsv(
    small_tsv_samples=350,
    init_tsv_pth="full_dataset.tsv",
    small_tsv_pth="subset_350.tsv"
)

# Evaluate COCO captions
metrics = compressed_data_scorer(
    predictions=pred_df,
    references=ref_df
)

Author: Thesis Research Project
Modified: For multimodal workload evaluation research
"""

import json
import shutil
import pandas as pd
import os
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from collections import Counter, defaultdict
import textwrap
from typing import Literal
# HuggingFace evaluate has rouge, sacrebleu
import evaluate
# CIDEr comes from coco-caption package
from pycocoevalcap.cider.cider import Cider
import cv2
import numpy as np
from scenedetect import open_video, SceneManager, ContentDetector
from typing import Set, List
from PIL import Image
from typing import Tuple
from decord import VideoReader, cpu
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# added zero for the default not-compressed data
PIXEL_REDUCTIONS = ["0", "5", "10", "20", "30", "40", "50", "60", "70", "80", "90"]

# number of rows I want for my small subset. I just have to change it and run to create the tsv file
SMALL_TSV_SAMPLES = 350
ROWS_CONTAIN = 350

# HOME_DATA_PATH = '/path/to/home'   # For local development
HOME_DATA_PATH = '/srv/muse-lab/datasets/VLMEvalKitdata'  # Shared server path


LMUData_PATH = os.path.join(HOME_DATA_PATH, 'LMUData')

CACHE_MODEL_DIR = "/srv/muse-lab/datasets/VLMEvalKitdata/.cache/huggingface"

MMBENCH_images = os.path.join(LMUData_PATH, 'images/MMBench')
COCO_VAL_images = os.path.join(LMUData_PATH, 'images/COCO')
LLaVABENCH_images = os.path.join(LMUData_PATH, 'images/LLaVABench')

CLEVER_SAMPLED_VIDEO_FRAMES_CACHE = os.path.join(LMUData_PATH, "clever_sampled_video_frames")

PLOTS_SAVE_PATH = "plots"
# Plots are saved relative to the current directory

# CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# # Go one directory up to get the project folder
# PTH_TO_PROJECT_FOLDER = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
# # OUTPUTS_FOLDER = os.path.join(PTH_TO_PROJECT_FOLDER, 'outputs')

OUTPUTS_FOLDER = os.path.join(HOME_DATA_PATH, 'outputs')

def save_figure_as_pdf(fig, plot_name):
    """
    Saves a matplotlib figure as a pdf to a specified directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be saved.
    plot_name : str
        The name of the plot to be saved.

    Notes
    -----
    If the directory does not exist, it will be created.
    If a file with the same name already exists, it will not be overwritten.
    """
    if not os.path.exists(PLOTS_SAVE_PATH):
        os.makedirs(PLOTS_SAVE_PATH)
    if not os.path.exists(os.path.join(PLOTS_SAVE_PATH, plot_name+".pdf")):
        print(f"Saving plot to {os.path.join(PLOTS_SAVE_PATH, plot_name+'.pdf')}")
        fig.savefig(os.path.join(PLOTS_SAVE_PATH, plot_name+".pdf"), bbox_inches='tight')

def bytes_converter(file_size_bytes, convert_to = "MB"):
    """
    Convert file size from bytes to megabytes (MB) with 3 decimal places.

    We divide by 1024 to convert bytes to KBs. We assume that KBs are 1024 bytes.
    
    Parameters:
        file_size_bytes (int or float): File size in bytes.
        
    Returns:
        float: File size in megabytes (MB) rounded to 3 decimal places.
    """
    if convert_to == "MB":
        megabytes = file_size_bytes / (1024 ** 2) # MBs
    else:
        megabytes = file_size_bytes / (1024 ** 1) # KBs
    # return megabytes
    return round(megabytes, 3)

def file_size(file_path, convert_to = "MB"):
    """
    this function will return the file size in MBs or KBs
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return bytes_converter(file_info.st_size, convert_to = convert_to)

def display_matplot_lib_img(pth):
    """
    Displays an image using matplotlib's imshow function and prints its size in KiloBytes.

    Parameters
    ----------
    pth : str
        The path to the image file to be displayed.

    Returns
    -------
    None
    """
    img = Image.open(pth)
    plt.imshow(img)
    plt.show()
    print("Image KiloBytes:", file_size(pth, convert_to = "KB"))    

def replace_video_ids_by_appearance(df, column_name="video", padding_length=3):
    """
    Replaces unique string identifiers in a DataFrame column with sequential,
    zero-padded numerical IDs, assigned based on their order of first appearance.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to process (default: "video").
        padding_length (int): The desired length for the new numerical IDs,
                              padded with leading zeros (e.g., 3 for '001', '010').

    Returns:
        pd.DataFrame: A new DataFrame with the specified column updated.
                      The original DataFrame is not modified.
    """
    df_copy = df.copy() # Create a copy to avoid modifying the original DataFrame

    # This dictionary will store the mapping from original_id to new_numerical_id
    id_mapping = {}
    
    # This counter will assign the next sequential number
    next_sequential_num = 1
    
    # This list will hold the new IDs in the order they should appear in the column
    new_ids_for_column = []

    # Iterate through the column values to build the mapping based on first appearance
    for original_id in df_copy[column_name]:
        if original_id not in id_mapping:
            # If this is the first time we see this video ID, assign a new sequential number
            new_padded_id = f"{next_sequential_num:0{padding_length}d}"
            id_mapping[original_id] = new_padded_id
            next_sequential_num += 1
        
        # Add the mapped ID (either newly assigned or already existing) to our list
        new_ids_for_column.append(id_mapping[original_id])
    
    # Assign the list of new IDs back to the DataFrame column
    df_copy[column_name] = new_ids_for_column
    return df_copy

def create_question_ids(df, video_col='video', question_id_col='question_id'):
    """
    Creates a new column with question IDs in the format 'VIDEOID-QUESTIONNUM',
    where QUESTIONNUM is sequential for each video.

    Args:
        df (pd.DataFrame): The input DataFrame, expected to have a 'video' column
                           with formatted IDs (e.g., '001', '002').
        video_col (str): The name of the column containing the video IDs.
        question_id_col (str): The name of the new column to create for question IDs.

    Returns:
        pd.DataFrame: A new DataFrame with the 'question_id' column added.
                      The original DataFrame is not modified.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # Step 1: Group by the 'video' column and calculate a cumulative count for each video.
    # .cumcount() generates a 0-indexed sequence (0, 1, 2...).
    # We add 1 to make it 1-indexed (1, 2, 3...).
    df_copy['temp_question_num'] = df_copy.groupby(video_col).cumcount() + 1

    # Step 2: Combine the video ID with the question sequence number.
    # Ensure both parts are treated as strings before concatenation.
    df_copy[question_id_col] = df_copy[video_col].astype(str) + '-' + \
                               df_copy['temp_question_num'].astype(str)

    # Step 3: Drop the temporary column used for sequencing
    df_copy = df_copy.drop(columns=['temp_question_num'])
    return df_copy

def display_last_element(element, tabs_num):
    # if it is string, don't print all of it and keep in mind to take care of the newline characters
    """
    Prints the last element in a nested list or dict, taking care of not printing too much if it is a string.

    If the element is a string, only the first 300 characters are printed and
    newline characters are handled properly. If the string is longer than 300
    characters, "..." is printed after the first 300 characters.

    Parameters
    ----------
    element : str or int or None
        The element to print.
    tabs_num : int
        The number of tabs to print before the element.

    """
    if isinstance(element, str):
        
        # print(tabs_num*"\t", element[:300].replace("\n", "\n"+tabs_num*"\t"))
        print("\n".join((tabs_num*"\t") + line for line in element[:300].splitlines()))

        if len(element) > 300:
            if len(element) > 300:
                print(tabs_num*"\t", "...")
    else:
        # int or None
        print(tabs_num*"\t", element)

def display_element(element, tabs_num=1):

    """
    Recursively prints the structure of a given element with indentation.

    The function handles elements that are dictionaries, lists, or other types.
    For dictionaries, it prints each key and recursively calls itself for the
    key's value. For lists, it prints all elements if the list contains fewer
    than 15 items, otherwise it only prints the first element. If the element
    is neither a dictionary nor a list, it calls `display_last_element` to
    handle the printing.

    Parameters
    ----------
    element : dict, list, or other types
        The element to be displayed. It can be a dict, list, int, None, or str.
    tabs_num : int, optional
        The number of tabs to print before the element, default is 1.

    Notes
    -----
      Handles the "image_size_pixel" key in a special way by printing its value directly.
      If the element is a dict, it will print the keys and the length of the values if they are lists.
      If the element is a list, it will print the first element and then recursively call itself on it.
      If the element is a string, it will print the first 300 characters and then "..." if it is longer than 300.
      If the element is an int or None, it will simply print it.
    """

    if isinstance(element, dict):
    
        for keys in element.keys():
            
            # Might be an inner key of data so check to print number of samples:
            if isinstance(element[keys], list):
                print(tabs_num*"\t", keys, " contains: ", len(element[keys]), sep='')
            else:
                print(tabs_num*"\t", keys, sep='')
            if keys != "image_size_pixel":
                display_element(element[keys], tabs_num+1)
            else:
                print((tabs_num+1)*"\t", element[keys], sep='')
    
    elif isinstance(element, list):
        # SOS
        # if they are a lot probably means they are a lot of samples so display only one
        if len(element) < 15:
            for list_item in element:
                display_element(list_item, tabs_num+1)
        else:
            display_element(element[0], tabs_num+1)
    else:
        display_last_element(element, tabs_num)

def universal_json_jsonl_printer(path_to_file):
    # Print the total path to the file
    """
    Prints the contents of a JSON or JSONL file, with indentation and prettiness.

    The function prints the total number of samples (lines) in the file.
    For each JSON object, it checks if any value is a list and prints
    the list elements and their keys. If not a list, it prints the key
    and value.

    Parameters
    ----------
    path_to_file : str
        The path to the JSON or JSONL file.

    Notes
    -----
        If the file is a JSONL file, it prints the total number of samples (lines) in the file.
        It then prints the first sample in a prettified format.
        If the file is a JSON file, it prints the contents of the file in a prettified format.
    """

    if not path_to_file.startswith("/"):
        if not path_to_file.startswith(".."):
            # print(110*"*", "\n", 110*"*", "\n", os.path.join(os.getcwd(), path_to_file), sep='')
            print(110*"*", "\n", os.path.join(os.getcwd(), path_to_file), sep='')
        else:
            # print(110*"*", "\n", 110*"*", "\n", 
            print(110*"*", "\n", 
                  os.path.join(
                      os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                      path_to_file[3:]
                      ),
                      sep='')
    else:
        # print(110*"*", "\n", 110*"*", "\n", path_to_file, sep='')
        print(110*"*", "\n", path_to_file, sep='')

    # Open the file and check if it is json or jsonl
    with open(path_to_file, "r") as f:
        
        if path_to_file.endswith(".jsonl"):
            
            data = f.readlines()
            print(f"File samples are {len(data)} in the format:")
            display_element(json.loads(data[0]))

        else:
            data = json.load(f)
            print(f"File samples are {len(data)} in the format:")
            display_element(data, 0)
            
        f.close()
    # print(110*"*", "\n", 110*"*", sep='')
    print(110*"*", sep='')

def read_n_ret_json(path_to_file):
    """
    Reads a JSON or JSONL file and returns its contents as a Python object.

    Parameters
    ----------
    path_to_file : str
        The path to the JSON or JSONL file.

    Returns
    -------
    data : object
        The contents of the file as a Python object.

    Notes
    -----
    If the file is a JSONL file, it reads the file line by line and returns a list of Python objects.
    If the file is a JSON file, it reads the file and returns a Python object.
    """
    with open(path_to_file, "r") as f:        
        data = json.load(f)
        f.close()
        return data

def jsonl_to_tsv(jsonl_filepath, tsv_filepath,
                 fieldnames = ['id', 'input', 'output', 'modality_path', 'modality_size']):
    """
    Converts a JSONL file to a TSV file.

    This function reads a JSONL file line by line, extracts specified fields
    from each JSON object, and writes them as rows in a TSV file. The fields
    to be extracted are defined in the `fieldnames` list. Each JSON object is
    expected to contain a 'request' dictionary from which the field values
    are obtained.

    Args:
        jsonl_filepath (str): The path to the input .jsonl file.
        tsv_filepath (str): The path where the output .tsv file will be saved.
        fieldnames (list of str, optional): A list of field names to extract
            from the JSON objects and use as TSV columns. Defaults to
            ['id', 'input', 'output', 'modality_path', 'modality_size'].
            Ensure this list matches the keys you expect within the 'request' object.

    Notes:
        - The TSV file is created with UTF-8 encoding and uses tab characters
          as delimiters.
        - Newline and tab characters in string values are replaced with spaces
          to ensure TSV format integrity.
        - If a field is missing in a JSON object, an empty string is used as
          the default value.
        - The function skips empty lines in the JSONL file and handles JSON
          decoding errors gracefully by printing a warning message.
    """
    
    # Open the TSV file in write mode, with newline='' to prevent extra blank rows
    # and specify UTF-8 encoding for broad character support.
    with open(tsv_filepath, 'w', newline='', encoding='utf-8') as tsvfile:
        # Create a csv.writer object, specifying '\t' as the delimiter for TSV
        writer = csv.writer(tsvfile, delimiter='\t')

        # Write the header row (column names)
        writer.writerow(fieldnames)

        # Open and read the JSONL file line by line
        with open(jsonl_filepath, 'r', encoding='utf-8') as jsonlfile:
            for line_num, line in enumerate(jsonlfile):
                # Skip empty lines
                if not line.strip():
                    continue

                try:
                    # Parse the JSON object from the current line
                    data = json.loads(line.strip())
                    # Extract the nested 'request' dictionary
                    request_data = data.get('request', {})

                    # Prepare a list to hold the data for the current TSV row
                    row_data = []
                    for field in fieldnames:
                        # Get the value for the current field, defaulting to an empty string
                        # if the key is not found in request_data
                        value = request_data.get(field, '')

                        # --- Data Sanitization for TSV ---
                        # Replace newline characters with spaces to ensure each TSV record
                        # remains on a single line.
                        # Replace tab characters to prevent breaking TSV column separation.
                        if isinstance(value, str):
                            value = value.replace('\n', ' ').replace('\t', ' ')
                        # If value is not a string (e.g., int, float), convert to string
                        # for writing to TSV.
                        else:
                            value = str(value)
                        # --- End Data Sanitization ---
                        row_data.append(value)

                    # Write the prepared row data to the TSV file
                    writer.writerow(row_data)

                except json.JSONDecodeError as e:
                    print(f"Warning: Error decoding JSON on line {line_num + 1}: {e}. Skipping line: '{line.strip()}'")
                except Exception as e:
                    print(f"Warning: An unexpected error occurred on line {line_num + 1}: {e}. Skipping line: '{line.strip()}'")

def calculate_accuracy_2_dataframe_columns(df, answer_col='answer', prediction_col='prediction'):
    """
    Calculates and prints the accuracy between two columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        answer_col (str): The name of the column containing the true answers.
        prediction_col (str): The name of the column containing the predictions.

    Returns:
        float: The calculated accuracy.
    """
    if answer_col not in df.columns:
        raise ValueError(f"'{answer_col}' column not found in DataFrame.")
    if prediction_col not in df.columns:
        raise ValueError(f"'{prediction_col}' column not found in DataFrame.")
    if df.empty:
        print("DataFrame is empty, accuracy cannot be calculated.")
        return 0.0

    # Compare the 'answer' and 'prediction' columns element-wise
    # This will result in a Series of True/False values
    # (True where they match, False where they don't)
    correct_predictions = (df[answer_col] == df[prediction_col])

    # Count the number of True values (which represent correct predictions)
    num_correct = correct_predictions.sum()

    # Get the total number of predictions
    total_predictions = len(df) # Or df.shape[0]
    # Calculate accuracy
    accuracy = num_correct / total_predictions
    # Print the result
    print(f"Accuracy: {num_correct} / {total_predictions} = {accuracy:.2%}")

    return accuracy

def delete_matching_items(folder_path, substrings, files_only=True, recurse_unmatched_dirs=False):
    """
    Deletes files (and optionally folders) in the specified folder if their name contains any of the given substrings.

    :param folder_path: Path to the folder to search in.
    :param substrings: List of substrings to check in file/folder names.
    :param files_only: If True, only deletes files. If False, deletes both files and folders.
    :param recurse_unmatched_dirs: If True, recursively enters subfolders that don't match and applies the same logic.
    """
    if not os.path.isdir(folder_path):
        print("Invalid folder path")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if any(sub in item for sub in substrings):
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
                elif not files_only and os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
        elif recurse_unmatched_dirs and os.path.isdir(item_path):
            # Recurse into subdirectory
            delete_matching_items(item_path, substrings, files_only, recurse_unmatched_dirs)

def read_csv_n_print(csv_pth, rows_print=3, sep = '\t', print_column_uniques=None):
    """
    Reads a CSV file and prints its shape and the first few rows. Optionally, prints the number of unique values for a specified column.

    Args:
        csv_pth (str): The file path to the CSV file.
        rows_print (int, optional): The number of rows to display from the top of the DataFrame. Defaults to 3.
        sep (str, optional): The delimiter used in the CSV file. Defaults to '\t'.
        print_column_uniques (str, optional): The column name for which to print the number of unique values. Defaults to None.

    Returns:
        None
    """

    data = pd.read_csv(csv_pth, sep = sep)
    print(data.shape)
    if print_column_uniques:
        print(data[print_column_uniques].nunique())
    display(data.head(rows_print))

def tsv_to_jsonl(tsv_filepath, jsonl_filepath, encoding='utf-8'):
    """
    Converts a TSV file to a JSONL file.
    The first row of the TSV is expected to be the header (column names).
    Each subsequent row will be converted into a JSON object, with column names
    as keys and row values as values. Each JSON object is written on a new line.

    Args:
        tsv_filepath (str): The path to the input .tsv file.
        jsonl_filepath (str): The path where the output .jsonl file will be saved.
        encoding (str): The character encoding for reading and writing files (default: 'utf-8').
    """
    if os.path.exists(jsonl_filepath):
        print('The jsonl file already exists.')
        return
    try:
        with open(tsv_filepath, 'r', newline='', encoding=encoding) as tsvfile:
            # Use csv.reader with tab delimiter
            reader = csv.reader(tsvfile, delimiter='\t')

            # Read the header row
            headers = next(reader)

            with open(jsonl_filepath, 'w', encoding=encoding) as jsonlfile:
                for row_num, row in enumerate(reader):
                    # Ensure the row has the same number of columns as headers
                    if len(row) != len(headers):
                        print(f"Warning: Row {row_num + 2} has {len(row)} columns, "
                              f"but {len(headers)} headers. Skipping this row: {row}")
                        continue

                    # Create a dictionary for the current row
                    row_dict = {}
                    for i, header in enumerate(headers):
                        # Optionally, try to convert numeric values
                        try:
                            # Attempt to convert to int if it looks like an integer
                            if row[i].strip().isdigit():
                                row_dict[header] = int(row[i])
                            # Attempt to convert to float if it looks like a float
                            elif row[i].strip().replace('.', '', 1).isdigit():
                                row_dict[header] = float(row[i])
                            else:
                                row_dict[header] = row[i]
                        except ValueError:
                            # If conversion fails, keep as string
                            row_dict[header] = row[i]

                    # For your specific JSONL example from previous turns,
                    # the keys were nested under a "request" object.
                    # If you want to replicate that specific nesting, you'd do:
                    # final_json_object = {"request": row_dict}
                    # But typically, a straight TSV to JSONL conversion makes
                    # each top-level column a key. Let's assume direct mapping.
                    final_json_object = row_dict

                    # Convert the dictionary to a JSON string and write to the file
                    jsonlfile.write(json.dumps(final_json_object) + '\n')

        print(f"Successfully converted '{tsv_filepath}' to '{jsonl_filepath}'")

    except FileNotFoundError:
        print(f"Error: The file '{tsv_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_n_show_tsv(jsonl_fil, tsv_fil, rows_to_contain=-1):    
    """
    Create a TSV file from a JSONL file and show the first few entries in the TSV file

    Parameters
    ----------
    jsonl_fil : str
        The JSONL file to convert
    tsv_fil : str
        The path to save the TSV file
    rows_to_contain : int
        The number of rows to include in the TSV file. If -1, all rows are included.
    """
    if not os.path.exists(tsv_fil):
        # Run the conversion function
        jsonl_to_tsv(jsonl_fil, tsv_fil)

        ################################################
        # make changes to the file
        data = pd.read_csv(tsv_fil, sep = '\t')
        data = data.reset_index()
        data.rename(columns={'input': 'question'}, inplace=True)
        data.rename(columns={'modality_path': 'image_path'}, inplace=True)
        data.rename(columns={'output': 'answer'}, inplace=True)
        # data.rename(columns={'id': 'index'}, inplace=True)
        ################################################
        if rows_to_contain > 0:
            data.head(rows_to_contain).to_csv(tsv_fil, sep="\t", index=False)
        else:
            data.to_csv(tsv_fil, sep="\t", index=False)
    else:
        # exists but check number of rows
        data = pd.read_csv(tsv_fil, sep = '\t')
        # number of rows
        if rows_to_contain > 0:
            if data.shape[0] != rows_to_contain:
                os.remove(tsv_fil)
                jsonl_to_tsv(jsonl_fil, tsv_fil)
                data = pd.read_csv(tsv_fil, sep = '\t')
                data.rename(columns={'input': 'question'}, inplace=True)
                data.rename(columns={'modality_path': 'image_path'}, inplace=True)
                data.rename(columns={'output': 'answer'}, inplace=True)
                # if rows_to_contain > 0:
                data.head(rows_to_contain).to_csv(tsv_fil, sep="\t", index=False)
                # else:
                #     data.to_csv(tsv_fil, sep="\t", index=False)

    data = pd.read_csv(tsv_fil, sep = '\t')
    print(data.shape)
    display(data.head(1))

def display_file_count(folder):
    """
    Prints the number of files in the given folder.

    Args:
        folder (str): The path to the folder whose file count will be displayed.

    Note:
        This function does not count subdirectories.
    """
    print(len(
    [name for name in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, name))]
    ))

def create_small_tsv(small_tsv_samples, init_tsv_pth, small_tsv_pth, unique_col_print="image", shuffle_data=False):
    """
    Creates a small tsv file containing a subset of the data from the initial tsv file.

    If the initial tsv file does not exist, it will be created from the small tsv file.
    If the initial tsv file exists, it will be checked if it has the correct number of rows.
    If not, the small tsv file will be deleted and recreated from the initial tsv file.

    Args:
        small_tsv_samples (int): The number of samples to include in the small tsv file.
        init_tsv_pth (str): The path to the initial tsv file.
        small_tsv_pth (str): The path to the small tsv file.
        unique_col_print (str): The column to print the number of unique values from.
        shuffle_data (bool): Whether to shuffle the data before taking the first samples.
    """
    # not exist, so create it
    first_time = not os.path.exists(init_tsv_pth)
    if not os.path.exists(init_tsv_pth):
        # the initial one will contain the full data and the small the subset
        data = pd.read_csv(small_tsv_pth, sep = '\t')
        big_copy = data.copy()
        print(big_copy.shape)
        #######################################
        # # remove the image column
        # big_copy.drop('image', axis=1, inplace=True)
        #######################################
        big_copy.to_csv(init_tsv_pth, sep="\t", index=False)

    # small one exists, check if it has the number of rows I want
    if os.path.exists(init_tsv_pth):
        # add an extra if for the case that I generally use shuffle and the tsv already exists
        if first_time:
            data = pd.read_csv(init_tsv_pth, sep = '\t')
            # number of rows
            if data.shape[0] != small_tsv_samples:
                # delete the small one and create a new one
                os.remove(small_tsv_pth)
                if shuffle_data:
                    # don't take only the first samples, but shuffle data
                    data = data.sample(n=small_tsv_samples, replace=False)
                    data.to_csv(small_tsv_pth, sep="\t", index=False)
                else:
                    data.head(small_tsv_samples).to_csv(small_tsv_pth, sep="\t", index=False)

    for i in [small_tsv_pth, init_tsv_pth]:
        read_csv_n_print(i, rows_print=1, sep = '\t', print_column_uniques=unique_col_print)

def delete_small_tsv_n_create_initial(init_pth, small_pth):
    """
    Deletes an existing small TSV file and creates a new one from an initial TSV file.
    
    Parameters
    ----------
    init_pth : str
        The file path to the initial TSV file, which contains the full dataset.
    small_pth : str
        The file path to the small TSV file that will be deleted and replaced with data from the initial TSV file.
    """

    if os.path.exists(init_pth):
        os.remove(small_pth)
        
        data = pd.read_csv(init_pth, sep = '\t')
        data.to_csv(small_pth, sep="\t", index=False)
        os.remove(init_pth)

def convert_column_values_to_correct_format(pth_to_tsv, column_name,
                                            dictionary_with_values_to_replace={
                                                0: 'A',
                                                1: 'B',
                                                2: 'C',
                                                3: 'D'
                                            }
                                            ):
    """
    Converts the values in a specified column of a TSV file to a different format based on a mapping dictionary.

    Parameters
    ----------
    pth_to_tsv : str
        The file path to the TSV file.
    column_name : str
        The name of the column in which the values are to be replaced.
    dictionary_with_values_to_replace : dict, optional
        A dictionary mapping current values to their replacements. Defaults to {0: 'A', 1: 'B', 2: 'C', 3: 'D'}.

    Notes
    -----
    If the column does not contain 'A' or 'B', it will apply the mapping and save the updated data back to the TSV file.
    """
    df = pd.read_csv(pth_to_tsv, sep='\t')
    if not 'A' in df[column_name].unique() or not 'B' in df[column_name].unique():
        df[column_name] = df[column_name].map(dictionary_with_values_to_replace)
        df.to_csv(pth_to_tsv, sep='\t', index=False)

def make_image_path_column(dataset_compr='MMBench_DEV_EN_bdp_lan_rgb_0',
                           images_folder=MMBENCH_images, verbose=False):
    """
    Adds a column to a TSV file with paths to images.
    
    Parameters
    ----------
    dataset_compr : str, optional
        The common prefix for the dataset TSV files. Defaults to 'MMBench_DEV_EN_bdp_lan_rgb_0'.
    images_folder : str, optional
        The path to the folder where the images are stored. Defaults to MMBENCH_images.
    verbose : bool, optional
        If True, prints the first row of the updated TSV file for each category. Defaults to False.
    
    Notes
    -----
    It first deletes the 'image' column if it exists, and then adds a new column 'image_path' with the paths to the images.
    For example, if the dataset name is 'MMBench_DEV_EN_bdp_lan_rgb_0', it will add a column 'image_path' with paths like 'path/MMBench_DEV_EN_bdp_lan_rgb_00/0.jpg'.
    """
    for num in NUMS:

        cur_cat = dataset_compr + num
        tsv_f = pd.read_csv(
            os.path.join(LMUData_PATH, cur_cat + '.tsv'),
            sep = '\t')
        print(cur_cat)

        if 'image' in tsv_f.columns: # has not been deleted yet
            tsv_f.drop(columns=['image'], inplace=True)
            tsv_f.to_csv(os.path.join(LMUData_PATH, cur_cat + '.tsv'), sep = '\t', index=False)
    
        if 'image_path' not in tsv_f.columns: # has not been added yet
            tsv_f['image_path'] = images_folder + '/' + cur_cat + '/' + tsv_f['index'].astype(str) + ".jpg"
            tsv_f.to_csv(os.path.join(LMUData_PATH, cur_cat + '.tsv'), sep = '\t', index=False)

        if verbose:        
            tsv_f = pd.read_csv(os.path.join(LMUData_PATH, cur_cat + '.tsv'), sep = '\t')
            display(tsv_f.head(1))

#########################################################################################
#########################################################################################
#########################################################################################
def resize_proportional(image: Image.Image, scale: int = 20) -> Image.Image:
        """
        Proportionally resizes an image by reducing both dimensions by `scale` percent.
        Ensures output dimensions are even numbers (rounded up).
        
        Parameters
        ----------
        image : PIL.Image
            Input image.
        scale : int
            Percentage reduction (e.g. 20 → shrink by 20%). 
            Can also be negative for enlargement (e.g. -20 → enlarge by 20%).
        """
        w, h = image.size

        # convert percentage into factor
        factor = 1 - (scale / 100.0)

        new_w = int(w * factor)
        new_h = int(h * factor)

        # round up to even numbers
        if new_w % 2 != 0:
            new_w += 1
        if new_h % 2 != 0:
            new_h += 1

        return image.resize((new_w, new_h), Image.LANCZOS)

def apply_clever_sampling(sampling_technique,
                          video_path: str,
                          max_frames: int = 32,
                          sampling_extra_params: int=None) -> np.ndarray:
    """
    Applies a clever sampling technique to a video and returns the sampled frames.
    
    Parameters
    ----------
    sampling_technique : str
        The technique to use for sampling. Supported values are 'scene_change', 'sharpness', and 'motion_based'.
    video_path : str
        The path to the video file.
    max_frames : int, optional
        The maximum number of frames to sample. Defaults to 32.
    sampling_extra_params : int, optional
        Extra parameters for the sampling technique. For example, the threshold for scene change detection.
    
    Returns
    -------
    frame_array : np.ndarray
    frame_array : np.ndarray
        The sampled frames.
        The sampled and resized frames as (N, H, W, C).
    frame_times : Any
        Placeholder for compatibility (fix if you want metadata).
    video_time : Any
        Placeholder for compatibility (fix if you want metadata).
    """
    ##########
    # folder that will contain the clever sampled frames, so they won't have to run again
    pth_to_saved_frames = CLEVER_SAMPLED_VIDEO_FRAMES_CACHE
    pth_to_sampling_techn = os.path.join(pth_to_saved_frames, sampling_technique)
    
    # /srv/muse-lab/datasets/VLMEvalKitdata/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/video/fFjv93ACGo8.mp4
    dataset_name = video_path.split("huggingface/hub/")[1].split("/snapshots")[0] # 'datasets--lmms-lab--Video-MME'
    pth_to_dataset = os.path.join(pth_to_sampling_techn, dataset_name)
    if "/video/" in video_path:
        video_name = video_path.split("/video/")[1].split(".mp4")[0] # 'fFjv93ACGo8'
    elif "/videos/" in video_path:
        video_name = video_path.split("/videos/")[1].split(".mp4")[0] # 'fFjv93ACGo8'
    else:
        raise ValueError(f"Could not parse video name from {video_path}")
    
    # instead of creating new npy specifically resized, return the original and reshape and save it outside
    # this way, I will have saved the general case and then resize outside for more specific cases
    pth_to_final_ar = os.path.join(pth_to_dataset, video_name + ".npy")
    
    if not os.path.exists(pth_to_dataset):
        os.makedirs(pth_to_dataset)
    else:
        if os.path.exists(pth_to_final_ar):
            return np.load(pth_to_final_ar), "NEEDS_FIXING", "NEEDS_FIXING"
    ##########

    # -----------------------------
    # Run sampling strategy
    # -----------------------------
    if sampling_technique == 'scene_change':
        # return sample_frames_by_scene_change_decord_version(video_path, max_frames=max_frames, threshold=sampling_extra_params)
        frame_array, _, _ = sample_frames_by_scene_change_decord_version(video_path, max_frames=max_frames, threshold=sampling_extra_params)
        # frame_array = sample_frames_by_scene_change(video_path, max_frames=max_frames, threshold=sampling_extra_params)
    elif sampling_technique == 'sharpness':
        frame_array = sample_frames_by_sharpness(video_path, max_frames=max_frames, sharpness_threshold=sampling_extra_params)
    elif sampling_technique == 'motion_based':
        frame_array = sample_frames_by_motion(video_path, max_frames=max_frames, motion_threshold=sampling_extra_params)
    else:
        raise ValueError(f"Unknown sampling technique: {sampling_technique}")
    
    # resizing happens outside IF necessary
    resized_frames = frame_array

    # Cache resized frames
    with open(pth_to_final_ar, 'wb') as f:
        np.save(f, resized_frames)
    return resized_frames, "NEEDS_FIXING", "NEEDS_FIXING"

def sample_frames_by_scene_change_decord_version(
    video_path: str,
    *,
    max_frames: int = 32,
    threshold: float = 27.0,
) -> Tuple[np.ndarray, List[str], float]:
    """
    The difference from sample_frames_by_scene_change_open_cv_version is that it
    returns some extra measurements, which are useful in VLMEvalKit and not in vllm
    statistics.

    Sample frames from a video based on scene changes using a content detector.

    This function performs scene detection on a given video file to identify
    distinct scenes by analyzing changes in content. It then selects key frames 
    from each identified scene and returns them. If no scenes are detected or 
    the number of selected frames exceeds the specified limit, frames are
    uniformly sampled from the video.

    Args:
        video_path (str): The path to the video file.
        max_frames (int, optional): The maximum number of frames to return. 
            Defaults to 32.
        threshold (float, optional): The threshold for the ContentDetector. 
            Higher values mean less sensitivity to changes, resulting in 
            fewer detected scene changes. Defaults to 27.0.

    Returns:
        Tuple[np.ndarray, List[str], float]: A tuple containing:
            - A numpy array of shape (N, H, W, 3) with the sampled frames.
            - A list of strings representing the timestamp of each frame in seconds.
            - The total duration of the video in seconds.
    """
    # -----------------------------
    # Step 1: Scene Detection
    # -----------------------------
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()  # List[Tuple[Timecode, Timecode]]

    idx: Set[int] = set()
    for start_tc, end_tc in scene_list:
        start, end = start_tc.get_frames(), end_tc.get_frames()
        idx.add(start)
        if end - start > 3:
            idx.add((start + end) // 2)
        if end - start >= 1:
            idx.add(end - 1)

    # -----------------------------
    # Step 2: Fallback or Downsample
    # -----------------------------
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps_val = vr.get_avg_fps()
    video_duration = total_frames / fps_val

    if not idx:
        idx = set(np.linspace(0, total_frames - 1, max_frames, dtype=int))

    if len(idx) > max_frames:
        sorted_idx = sorted(idx)
        keep = np.linspace(0, len(sorted_idx) - 1, max_frames, dtype=int)
        idx = {sorted_idx[i] for i in keep}

    sorted_idx = sorted(idx)

    # -----------------------------
    # Step 3: Extract Frames and Timestamps
    # -----------------------------
    frames_np = vr.get_batch(sorted_idx).asnumpy()  # (N, H, W, 3), RGB
    frame_times = [f"{i / fps_val:.2f}s" for i in sorted_idx]

    return frames_np, frame_times, video_duration

# initial version from mllm
def sample_frames_by_scene_change_open_cv_version(
    video_path: str,
    *,
    max_frames: int = 32,
    threshold: float = 27.0,
) -> np.ndarray:
    """
    The difference from the sample_frames_by_scene_change_decord_version is that it doesn't count
    timestamps and duration. It takes shorter amount of time.

    Sample `max_frames` frames from `video_path` by first detecting scene cuts and
    then selecting (1) the first, (2) middle, and (3) last frame of every scene.
    Falls back to uniform sampling when no scenes are found.

    Scenes are detected using the ContentDetector from scenedetect, which uses a threshold
    of 27.0 by default. The first frame of each scene is selected. If the scene is long
    enough, the middle frame is also selected. The last frame of each scene is selected
    if the scene is long enough

    Concept: Identify moments where the visual content changes significantly (scene cuts).
    Sample one or more frames from each distinct scene

    Parameters
    ----------
    video_path : str
        Path to the input video.
    max_frames : int, default 32
        Hard cap on the number of frames returned.
    threshold : float, default 27.0
        Sensitivity of the cut detector.  ↓ threshold  ⇒ more cuts detected.

    Returns
    -------
    np.ndarray
        Array of RGB frames with shape (N, H, W, 3), where N ≤ `max_frames` containing the sampled frames
    """
    # ------------------------------------------------------------------
    # 1. Scene-detect quickly on down-scaled frames
    # ------------------------------------------------------------------
    video = open_video(video_path)
    scene_manager = SceneManager()
    # ContentDetector: detects fast cuts using weighted average of HSV change. The ContentDetector
    # works by comparing successive frames of a video. If difference <= threshold, the frames
    # are considered part of the same scene. Higher threshold: Fewer scene changes will be
    # detected. The detector will be less sensitive to minor changes and will only mark very distinct cuts
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()  # (start_tc, end_tc) tuples
    # ------------------------------------------------------------------
    # 2. Pick candidate indices: first / mid / last of each scene
    # ------------------------------------------------------------------
    idx: Set[int] = set()
    for start_tc, end_tc in scene_list:
        start, end = start_tc.get_frames(), end_tc.get_frames()          # end is exclusive
        idx.add(start)                                                   # first
        if end - start > 3:                                              # middle
            idx.add((start + end) // 2)
        if end - start >= 1:                                             # last in scene
            idx.add(end - 1)
    # ------------------------------------------------------------------
    # 3. Thin if necessary – uniformly across the *already* selected idx
    # ------------------------------------------------------------------
    if len(idx) > max_frames:
        sorted_idx = sorted(idx)
        keep = np.linspace(0, len(sorted_idx) - 1, max_frames, dtype=int)
        idx = {sorted_idx[i] for i in keep}
    # ------------------------------------------------------------------
    # 4. Fallback: no scenes (or empty idx) → uniform sampling
    # ------------------------------------------------------------------
    # Maybe can be improved
    if not idx:
        # total = int(video.props.num_frames)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx = set(np.linspace(0, total - 1, max_frames, dtype=int))

    sorted_idx: List[int] = sorted(idx)
    # ------------------------------------------------------------------
    # 5. Grab frames efficiently (single forward scan rather than random seeks)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    next_target = 0
    target_count = len(sorted_idx)

    for frame_no in range(sorted_idx[-1] + 1):          # iterate once
        ok, frame = cap.read()
        if not ok:
            break                                       # EOF / corruption
        if frame_no == sorted_idx[next_target]:
            frames.append(frame[..., ::-1])             # BGR→RGB if needed
            next_target += 1
            if next_target == target_count:             # collected all
                break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return np.stack(frames)

def _thin_out_frames(selected_indices: list[int], max_frames: int) -> list[int]:
    """
    If a selection strategy yields too many frames, this function thins them out
    uniformly to a maximum number of frames. Ensures unique and sorted indices.

    Args:
        selected_indices (list[int]): A list of frame indices identified by a strategy.
        max_frames (int): The maximum number of frames desired in the output.

    Returns:
        list[int]: A new list of unique, sorted, and thinned frame indices.
    """
    unique_sorted_indices = sorted(list(set(selected_indices)))

    if len(unique_sorted_indices) <= max_frames:
        return unique_sorted_indices

    # Use linspace to select evenly spaced indices from the already selected ones
    # np.round is used to ensure we get integer indices from the float array
    thinning_indices_float = np.linspace(0, len(unique_sorted_indices) - 1, max_frames)
    thinning_indices = np.round(thinning_indices_float).astype(int)
    
    return [unique_sorted_indices[i] for i in thinning_indices]

# def _read_and_uniformly_sample_frames(path: str, max_frames: int = -1) -> npt.NDArray:
def _read_and_uniformly_sample_frames(path: str, max_frames: int = -1) -> np.ndarray:
    """
    Reads all frames from a video and then uniformly samples a specified number of frames.
    If max_frames is -1 or greater than total frames, all frames are returned.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise ValueError(f"No frames read from video file {path}.")
    
    frames_np = np.stack(all_frames)
    total_frames = frames_np.shape[0]

    if max_frames == -1 or max_frames >= total_frames:
        return frames_np
    
    # Generate integer indices evenly spaced from 0 to total_frames - 1
    # np.linspace(start, stop, num) generates num evenly spaced samples over the interval [start, stop]
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    # frames[frame_indices, :, :, :] is equivalent to frames[frame_indices, ...]
    sampled_frames = frames_np[frame_indices, ...]
    
    # A sanity check, though linspace with dtype=int should usually produce max_frames elements
    if len(sampled_frames) < max_frames:
        print(f"Warning: Expected {max_frames} frames but got {len(sampled_frames)} after uniform sampling. Video might be too short.")

    return sampled_frames

# def sample_frames_by_sharpness(video_path: str, max_frames: int = 32, sharpness_threshold: float = 100.0) -> npt.NDArray:
def sample_frames_by_sharpness(video_path: str, max_frames: int = 32, sharpness_threshold: float = 100.0) -> np.ndarray:
    """
    Concept: Discard blurry or very low-quality frames, even if they are uniformly sampled. Prioritize
    sharper frames.

    Samples frames based on their sharpness, prioritizing sharper frames. Sharpness is measured
    using the variance of the Laplacian operator, which is sensitive to edges and high-frequency
    content. If too many sharp frames are found, they are uniformly thinned out

    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to return.
        sharpness_threshold (float): Minimum Laplacian variance for a frame to be
                                     considered "sharp enough" to be selected.
                                     Higher values mean only very sharp frames are selected.
                                     Typical values vary widely depending on resolution
                                     and content (e.g., 50 for blurry, 500+ for very sharp).
                                     This parameter often requires tuning.

    Returns:
        npt.NDArray: A NumPy array of sampled frames.
    
    Raises:
        ValueError: If video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    selected_indices = []
    
    current_frame_idx = 0
    # print(f"  Analyzing sharpness for {video_path} (threshold={sharpness_threshold})...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Convert to grayscale, as Laplacian works on single-channel images
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian operator
        # cv2.CV_64F is used as the depth of the output image to avoid overflow
        # when computing variance, as Laplacian can produce negative values.
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        
        # The variance of the Laplacian indicates the amount of edges and thus sharpness.
        sharpness_score = laplacian.var()
        
        # Select frame if its sharpness score exceeds the threshold
        if sharpness_score > sharpness_threshold:
            selected_indices.append(current_frame_idx)
            
        current_frame_idx += 1
        
    cap.release()

    if not selected_indices:
        print(f"  Warning: No frames selected by sharpness for {video_path}. Falling back to uniform sampling.")
        # Fallback to uniform sampling if no sharp frames are found or threshold is too high
        return _read_and_uniformly_sample_frames(video_path, max_frames=max_frames)

    # print(f"Selected {len(selected_indices)} frames based on sharpness. Thinning to {max_frames} if necessary.")
    final_indices = _thin_out_frames(selected_indices, max_frames)  # Thin out selected frames if too many
    
    # Read only the selected frames using OpenCV for efficiency
    sampled_frames = []
    cap = cv2.VideoCapture(video_path) # Reopen for specific frame access
    if not cap.isOpened():
        raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

    for idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"  Warning: Could not read frame at index {idx} from {video_path}. Skipping.")
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were successfully sampled from {video_path} using sharpness detection.")

    return np.stack(sampled_frames)

def sample_frames_by_motion(video_path: str, max_frames: int = 32, motion_threshold: float = 1.0) -> np.ndarray:
    """
    Concept: Prioritize frames where significant motion or relevant activity occurs, assuming these
    frames carry more information about actions.

    How: Calculate motion vectors (e.g., optical flow) between frames. Frames with high optical flow
    magnitude could indicate important action.

    Samples frames based on the amount of motion detected between consecutive frames using
    dense optical flow (Farneback method). Frames with motion exceeding a `motion_threshold`
    are prioritized. If too many frames are selected, they are uniformly thinned out.

    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to return.
        motion_threshold (float): Average optical flow magnitude threshold. A frame is
                                  considered "active" if the mean magnitude of motion
                                  vectors between it and the previous frame exceeds this value.
                                  Typical values range from 0.1 (very sensitive) to 5.0 (only
                                  major movements). This parameter often requires tuning.

    Returns:
        npt.NDArray: A NumPy array of sampled frames.
    
    Raises:
        ValueError: If video cannot be opened or first frame cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    ret, prev_frame_bgr = cap.read()    # Read the first frame
    if not ret:
        cap.release()
        raise ValueError(f"Could not read first frame from {video_path}.")
    
    prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    selected_indices = [0] # Always include the first frame as a starting point
    
    current_frame_idx = 1
    while True:
        ret, next_frame_bgr = cap.read()
        if not ret:
            break # End of video
        
        next_frame_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate Farneback optical flow (dense flow)
        # Parameters for Farneback (tuned for general use, can be adjusted):
        # pyr_scale=0.5: pyramid scale, reduces image size by half at each level
        # levels=3: number of pyramid layers
        # winsize=15: averaging window size; larger -> smoother motion, less noise, more computation
        # iterations=3: number of iterations at each pyramid level
        # poly_n=5, poly_sigma=1.2: polynomial expansion size and standard deviation for Gaussian
        # flags=0: typically 0 for standard optical flow (or cv2.OPTFLOW_FARNEBACK_GAUSSIAN for Gaussian window)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, 
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the magnitude (length) of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate the average motion magnitude across the entire frame
        avg_motion = np.mean(magnitude)
        
        # Select frame if average motion exceeds the threshold
        if avg_motion > motion_threshold:
            selected_indices.append(current_frame_idx)
            
        prev_frame_gray = next_frame_gray # Update previous frame for next iteration
        current_frame_idx += 1
        
    cap.release()

    if len(selected_indices) <= 1: # Only the initial frame was selected, or no significant motion
        print(f"  Warning: Few or no frames selected by motion for {video_path}. Falling back to uniform sampling.")
        # Fallback to uniform if no motion detected or threshold is too high
        return _read_and_uniformly_sample_frames(video_path, max_frames=max_frames)

    # print(f"Selected {len(selected_indices)} frames based on motion. Thinning to {max_frames} if necessary.")
    # Thin out selected frames if too many
    final_indices = _thin_out_frames(selected_indices, max_frames)
    
    # Read only the selected frames using OpenCV for efficiency
    sampled_frames = []
    cap = cv2.VideoCapture(video_path) # Reopen cap for specific frame access
    if not cap.isOpened():
        raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

    for idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"  Warning: Could not read frame at index {idx} from {video_path}. Skipping.")
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were successfully sampled from {video_path} using motion detection.")

    return np.stack(sampled_frames)

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# plots
param_dictionary ={
    "title_size": 26.4,
    # "figsize_mul": 1.13,
    "figsize_mul": 2,
    "params_label_size": 21.5,
    "legend_font_size": 19,
    "label_size": 26,
    "plot_line_width": 2.72,
    "plot_marker_size": 11.5,
    "move_y_title_label": 0.42,
    "ymax_value": None,
    "ymin_value": 0.0,
    "mul_col_size": 6.45,
    "mul_row_size": 5.45,
    "x_ticks": [1, 2, 3, 4, 5],
    "y_ticks": None,
    "xmin_value": None,
    "axis_title_size": 28,
    "legend_size": 19,
    # "xlabel_size": 23,
    "xlabel_size": 29,
    # "ylabel_size": 23,
    "ylabel_size": 29,
    "y_params_label_size": 26,
    "x_params_label_size": 26,
    # "line_width": 2.3,
    "line_width": 4,
    "bar_width": 0.81
}

plot_colors_by_num = {
    0: "#e6194b",  # red
    1: "#3cb44b",  # green
    2: "#ffe119",  # yellow
    3: "#4363d8",  # blue
    4: "#f58231",  # orange
    5: "#911eb4",  # purple
    6: "#46f0f0",  # cyan
    7: "#f032e6",  # magenta
    8: "#bcf60c",  # lime
    9: "#fabebe",  # pink
    10: "#008080", # teal
    11: "#e6beff", # lavender
    12: "#9a6324", # brown
    13: "#fffac8", # light yellow
    14: "#800000", # maroon
}

def get_colormap_colors(n, cmap_name="Blues", vmin=0.2, vmax=0.8):
    """
    Generate `n` colors from a truncated colormap that avoids fading to white.
    
    Parameters
    ----------
    n : int
        Number of colors to generate.
    cmap_name : str
        Name of the matplotlib colormap (e.g., "Blues", "Greens", "Reds").
    vmin, vmax : float
        Range within the colormap to sample (0.0-1.0). 
        Lower values = darker, higher values = lighter.
    
    Returns
    -------
    colors : list
        List of RGBA colors.
    """
    # base_cmap = cm.get_cmap(cmap_name)
    base_cmap = plt.colormaps.get_cmap(cmap_name)
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base_cmap(np.linspace(vmin, vmax, 256))
    )
    return [truncated_cmap(i) for i in np.linspace(0, 1, n)]

def acc_pixel_red(accs, original_accuracy, y_axis="Accuracy (%)", round_up=True,
                  title=None, x_ziper=PIXEL_REDUCTIONS, x_label_tit="Pixel Reduction (%)"):
    """
    Plot the accuracy of a model under different pixel reductions.

    Parameters
    ----------
    accs : list
        A list of accuracy values, one for each pixel reduction.
    original_accuracy : float
        The accuracy of the model without any pixel reduction.
    y_axis : str, optional
        The label for the y-axis. Defaults to "Accuracy (%)".
    round_up : bool or str, optional
        Whether to round up the y-axis values to the nearest integer or not.
        If set to "int", will round up to the nearest integer.
        If set to True, will round up to two decimal places.
        If set to False, will keep the original value.
        Defaults to True.
    title : str, optional
        The title of the plot. Defaults to None.
    x_ziper : list, optional
        The list of pixel reductions to use. Defaults to PIXEL_REDUCTIONS.
    x_label_tit : str, optional
        The label for the x-axis. Defaults to "Pixel Reduction (%)".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object of the plot.
    """
    x_zip = range(len(x_ziper))

    xlabel_size = 29
    ylabel_size = 29
    x_params_label_size = 24.5
    y_params_label_size = 24.5

    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    # Wrap the string to a desired width (e.g., 20 characters)
    wrapped_label = textwrap.fill(y_axis, width=20)
    # Now, use the wrapped label in your plot
    ax.set_ylabel(wrapped_label, fontsize=ylabel_size)

    # # Pick a base colormap (Blues, Reds, Oranges, Greens, etc.)
    # cmap = plt.cm.Blues # gives you a gradient in one color family. You can swap with Reds, Oranges, Greens, or even continuous colormaps like viridis.
    # # Normalize positions so colors go from dark to light
    # norm = plt.Normalize(vmin=0, vmax=len(x_zip)-1)
    accs.insert(0, original_accuracy)
    # for idx, (x, y) in enumerate(zip(x_zip, accs)):
    #     color = plot_colors_by_num[idx % len(plot_colors_by_num) + 1]   # + 1 so that it is the same with the KV caches
    #     ax.bar(x, y, color=color, label=x_ziper[idx])
    colors = get_colormap_colors(len(x_zip), cmap_name="Blues", vmin=0.2, vmax=0.7)
    ax.bar(x_zip, accs, color=colors)
    
    ax.set_xticks(x_zip)
    ax.tick_params(axis='x', labelsize=x_params_label_size, rotation=45)
    ax.set_xticklabels(
        [perc for perc in x_ziper],
        rotation=45,  # Optional: angle labels if they overlap
        # ha='right'    # Optional: align labels for better spacing
    )

    ax.tick_params(axis='y', labelsize=y_params_label_size)

    if round_up:
        if round_up == "int":
            formatter = FuncFormatter(lambda x, _: f"{int(x * 100)}")
        else:
            formatter = FuncFormatter(lambda x, _: f"{x:.2f}")
    else:
        # keep some decimals instead of casting to int
        formatter = FuncFormatter(lambda x, _: f"{x:.2e}")  # scientific notation
        ax.yaxis.set_major_formatter(formatter)

    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel(x_label_tit, fontsize=xlabel_size)

    ax.axhline(
        y=original_accuracy,
        color="black",
        linestyle="--",
        linewidth=param_dictionary["line_width"],
        xmin=0, xmax=1  # full width of the axes
    )

    if title:
        ax.set_title(title, fontsize=param_dictionary["title_size"])
    
    plt.show()
    return fig

###################################################
NUMS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
techniques_list = ['scene_change', 'sharpness', 'motion_based']
max_frames_list=[4, 8, 16, 32, 64]
sampling_params_list=[
    [10, 20, 27, 34, 44], # scene_change
    [50, 100, 200, 500], # sharpness
    [1], # motion_based
    ]
###################################################
models = [
    "Qwen2-VL-2B-Instruct",
    # "Qwen2-VL-7B-Instruct",
    # "llava_onevision_qwen2_7b_ov",
    # "llava_onevision_qwen2_0.5b_ov",
    # "Pixtral-12B",
]
datasets = [
    "MMBench_DEV_EN",
    # "COCO_VAL",
    # "LLaVABench",
    # "Video-MME_64frame",
    # "MMBench_Video_64frame_nopack",
    # "TempCompass_Captioning_64frame"
]
MMBench_Video_64frame_nopack_tmp_folders = {
    "Qwen2-VL-7B-Instruct": "T20250919_G6778891c",
    "llava_onevision_qwen2_7b_ov": "T20250827_G4df0d32c",
    "llava_onevision_qwen2_0.5b_ov": "T20250826_Gebf8e8c1",
    "Qwen2-VL-2B-Instruct": "T20250919_G6778891c",
    "Pixtral-12B": "T20250917_G6778891c",
}
MMBench_Video_64frame_nopack_tmp_folders_scene_change = {
    "llava_onevision_qwen2_7b_ov": "T20250901_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250901_G9155fa72",
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "Pixtral-12B": "T20250917_G6778891c",
}
MMBench_Video_64frame_nopack_tmp_folders_sharpness = {
    "llava_onevision_qwen2_7b_ov": "T20250901_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250901_G9155fa72",
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "Pixtral-12B": "T20250917_G6778891c",
}
MMBench_Video_64frame_nopack_tmp_folders_motion_based = {
    "llava_onevision_qwen2_7b_ov": "T20250830_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250906_G59665fa1",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "Pixtral-12B": "T20250917_G6778891c",
}
TempCompass_Captioning_64frame_tmp_folders = {
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "llava_onevision_qwen2_7b_ov": "T20250901_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250901_G9155fa72",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "Pixtral-12B": "T20250917_G6778891c",
}
TempCompass_Captioning_64frame_tmp_folders_sharpness = {
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "llava_onevision_qwen2_7b_ov": "T20250902_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250902_G9155fa72",
    "Pixtral-12B": "T20250917_G6778891c",
}
TempCompass_Captioning_64frame_tmp_folders_scene_change = {
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "llava_onevision_qwen2_7b_ov": "T20250902_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250902_G9155fa72",
    "Pixtral-12B": "T20250919_G6778891c",
}
TempCompass_Captioning_64frame_tmp_folders_motion_based = {
    "llava_onevision_qwen2_7b_ov": "T20250904_G9155fa72",
    "llava_onevision_qwen2_0.5b_ov": "T20250908_G9bb54abe",
    "Qwen2-VL-7B-Instruct": "T20250918_G6778891c",
    "Qwen2-VL-2B-Instruct": "T20250918_G6778891c",
    "Pixtral-12B": "T20250919_G6778891c",
}

def coco_visualizer(mod, dat, coco_score=None):
    """
    Visualizes the COCO results for a given model and dataset.

    Args:
    - mod (str): Model name.
    - dat (str): Dataset name.
    - coco_score (list or None): If not None, a list containing the name of the COCO score
        and the original accuracy to compare with. If None, it will just print out all
        the accuracy scores of COCO.

    Returns:
    - None
    """
    accs = []
    if coco_score: # I want to visualize a specific score about COCO
        score_name = coco_score[0]
        orig_acc = coco_score[1]

        for num in NUMS:
            pth = os.path.join(OUTPUTS_FOLDER, mod, mod + "_" + dat + '_bdp_lan_rgb_0' + num + '_score.json')
            if os.path.exists(pth):
                # universal_json_jsonl_printer(pth)
                json_data = read_n_ret_json(pth)
                if not score_name.startswith("Bleu"):
                    score_compr = json_data[score_name]
                else:
                    blue_num = int(score_name.split("_")[-1])
                    score_compr = json_data["Bleu"][blue_num]
                accs.append(score_compr)
        print(accs)
        fig = acc_pixel_red(accs, orig_acc, round_up=False, title=mod, y_axis=score_name)
        save_figure_as_pdf(fig, mod + "_" + dat + "_coco_" + score_name)

def LLaVABench_score(data):
    """
    Compute the relative score between VLM and GPT4 for LLaVABench evaluation.

    Args:
    - data (pd.DataFrame): The dataframe containing the evaluation results, with columns 'score' and 'gpt4_score'.

    Returns:
    - pd.DataFrame: A dataframe containing the relative score between VLM and GPT4 for each category.
    """
    cates = ['overall'] + list(set(data['category']))
    ret = defaultdict(list)
    for c in cates:
        ret['split'].append(c)
        sub = data[data['category'] == c] if c != 'overall' else data
        ret['Relative Score (main)'].append(np.mean(sub['score']) / np.mean(sub['gpt4_score']) * 100)
        ret['VLM Score'].append(np.mean(sub['score']) * 10)
        ret['GPT4 Score'].append(np.mean(sub['gpt4_score']) * 10)
    return pd.DataFrame(ret)

def extract_scores(text):
    """Extract two numeric scores from a text review string."""
    if pd.isna(text):
        return (np.nan, np.nan)

    # 1. Match the "/10" style explicitly
    match = re.findall(r'\b(\d+)\s*/\s*10\b', text)
    if len(match) >= 2:
        return (float(match[0]), float(match[1]))
    elif len(match) == 1:
        return (float(match[0]), np.nan)

    # 2. Match standalone numbers, but avoid "Assistant 1", "Model 2" etc.
    match = re.findall(r'(?<!Assistant\s)(?<!Model\s)\b\d+\b', text, flags=re.IGNORECASE)
    if len(match) >= 2:
        return (float(match[0]), float(match[1]))
    elif len(match) == 1:
        return (float(match[0]), np.nan)

    # 3. Nothing found
    return (np.nan, np.nan)

def get_random_rows(df, n, random_state=42):
    """
    Selects n random rows from a pandas DataFrame.

    :param df: The input DataFrame.
    :param n: The number of random rows to select.
    :return: A new DataFrame with n random rows.
    """
    if n > len(df):
        raise ValueError("n cannot be greater than the number of rows in the DataFrame.")
    
    # set random seed
    return df.sample(n=n, random_state=random_state)

def llavabench_fix_bad_reviews(excel_csv):
    """Fix bad reviews and compute scores for LLaVABench."""
    not_none_indices = excel_csv[excel_csv['fail_review'].notna()].index.tolist()
    if len(not_none_indices) == 0:
        return LLaVABench_score(excel_csv), excel_csv

    # Copy rows with reviews
    result_df = excel_csv.loc[not_none_indices].copy()

    # Remove rows where fail_review is float
    result_df = result_df[~result_df['fail_review'].apply(lambda x: isinstance(x, float))].copy()

    # Extract (gpt4_score, score) safely
    scores = result_df['fail_review'].apply(lambda x: pd.Series(extract_scores(x)))
    scores.columns = ['gpt4_score', 'score']

    # Merge results back into result_df
    result_df = result_df.assign(**scores)

    # Update original dataframe (only these indices!)
    excel_csv.loc[result_df.index, ['gpt4_score', 'score']] = result_df[['gpt4_score', 'score']].astype(float)

    # at this point, some are still -1 or None because there are values like fail_review==8. Remove those rows
    excel_csv = excel_csv[~((excel_csv['gpt4_score'] == -1) | (excel_csv['gpt4_score'].isna()) | (excel_csv['score'] == -1) | (excel_csv['score'].isna()))]

    # # --- Display context ---
    # all_indices_to_display = set(not_none_indices)
    # max_index = excel_csv.index.max()
    # for idx in not_none_indices:
    #     if idx > excel_csv.index.min():
    #         all_indices_to_display.add(idx - 1)
    #     if idx < max_index:
    #         all_indices_to_display.add(idx + 1)
    # sorted_indices = sorted(list(all_indices_to_display))
    # print("\nDisplaying bad lines with context:")
    # display(excel_csv.loc[sorted_indices])

    results_df = LLaVABench_score(excel_csv)
    return results_df, excel_csv

def calculate_accuracy_on_MMBench_DEV_EN(pth):
    excel = pd.read_excel(pth)
    excel['cleaned_prediction'] = excel['prediction'].str.extract(r'\b([A-D])\b', flags=re.IGNORECASE)
    acc = calculate_accuracy_2_dataframe_columns(excel, answer_col='answer', prediction_col='cleaned_prediction')
    return acc

def compressed_data_scorer(mod, dat, someprints=None, type="COCO", orig_acc=None, extra_list=None, only_return_accs=False):
    """
    Compute accuracy scores for a given model and dataset over a set of compressed data.

    Parameters
    ----------
    mod : str
        Model name.
    dat : str
        Dataset name.
    someprints : list or None
        List of nums to print. If None, print all.
    type : str
        Type of the benchmark. Can be "COCO", "MMBench_DEV_EN", or "LLaVABench".
    orig_acc : float
        Original accuracy score to compare with.
    extra_list : list or None
        Extra list of nums to print. Mostly list with scores from LLaVABench original gpt4scores or something
    """
    print("-" * 35, " COMPRESSED ", "-" * 35, sep='')
    accs = []
    y_axis = "Accuracy (%)"

    ret_orig_llavabench = []
    if type == "LLaVABench":
        # To compute and display Other Metrics for LLaVABench
        pth = os.path.join(OUTPUTS_FOLDER, mod, mod + "_" + dat + '_openai_result.xlsx')
        excel = pd.read_excel(pth)
        _, renewedcsv = llavabench_fix_bad_reviews(excel)

        # To compute and display Other Metrics for LLaVABench
        for score_current in ["rouge", "bleu", "cider"]:
            ret_orig_llavabench.append(
                evaluate_llm_scores(
                    df=renewedcsv,
                    ref_col="caption",
                    pred_col="prediction",
                    score_type = score_current
                )
            )
        
        for i in ret_orig_llavabench:
            print(i)

    # To compute and display Other Metrics for LLaVABench
    llavabench_static_metrics = []
    for idx, num in enumerate(NUMS):
        if (someprints and num in someprints) or not someprints:
            if type == "COCO":
                pth = os.path.join(OUTPUTS_FOLDER, mod, mod + "_" + dat + '_bdp_lan_rgb_0' + num + '_score.json')
                # if os.path.exists(pth):
                #     universal_json_jsonl_printer(pth)
            elif type == "MMBench_DEV_EN":
                pth = os.path.join(OUTPUTS_FOLDER, mod, mod + "_" + dat + '_bdp_lan_rgb_0' + num + '.xlsx')
                if os.path.exists(pth):
                    # excel = pd.read_excel(pth)
                    # excel['cleaned_prediction'] = excel['prediction'].str.extract(r'\b([A-D])\b', flags=re.IGNORECASE)
                    # acc = calculate_accuracy_2_dataframe_columns(excel, answer_col='answer', prediction_col='cleaned_prediction')
                    acc = calculate_accuracy_on_MMBench_DEV_EN(pth)
                    accs.append(acc)
                round_up = "int"
            elif type == "LLaVABench":
                pth = os.path.join(OUTPUTS_FOLDER, mod, mod + "_" + dat + '_bdp_lan_rgb_0' + num + '_score.csv')
                
                if os.path.exists(pth):
                    y_axis = "Overall Relative Score"
                    pth_excel = os.path.join(OUTPUTS_FOLDER, mod, mod + "_" + dat + '_bdp_lan_rgb_0' + num + '_openai_result.xlsx')
                    excel_reviews = pd.read_excel(pth_excel)
                    renewed_scores, renewed_excel_csv = llavabench_fix_bad_reviews(excel_reviews)
                    if renewed_scores is not None:
                        final_scores = renewed_scores.copy()
                    else:
                        final_scores = pd.read_csv(pth)
                    
                    return_scores = renewed_excel_csv["score"] # "gpt4_score"
                    return_scores = return_scores.tolist()

                    # remove nan
                    return_scores = [x for x in return_scores if not math.isnan(x)]
                    ret_score = final_scores["Relative Score (main)"]
                    # get row overall
                    ret_score_overall = ret_score[final_scores["split"] == "overall"].item()
                    accs.append(ret_score_overall)
                    round_up = "no_multiply"

                    # also plot LLaVABench scores CDFs
                    if extra_list is not None and num=="0":
                    # if extra_list is not None:
                        plot_simple_cdf([extra_list, return_scores], x_ax_tit="LLaVABench score", 
                                        names=[
                                            "LLaVABench original scores", "LLaVABench scores " + PIXEL_REDUCTIONS[idx]
                                            ],
                                        dataset_name=dat)
                    list_scores = []
                    for score_current in ["rouge", "bleu", "cider"]:
                        list_scores.append(
                            evaluate_llm_scores(
                                df=renewed_excel_csv,
                                ref_col="caption",
                                pred_col="prediction",
                                score_type = score_current
                            )
                        )
                    llavabench_static_metrics.append(list_scores)
    
    if only_return_accs:
        return accs
    # To compute and display Other Metrics for LLaVABench
    if len(ret_orig_llavabench):

        for metric_idx, metric_nam, tit_score in [
            [0, "rouge1", " - Overlap of unigrams"],
            [0, "rouge2", " - Overlap of bigrams"],
            [0, "rougeL", " - Longest Common Subsequence (LCS)"],
            [0, "rougeLsum", " - sentence-level LCS"],
            [1, "score", " - BLEU score"],
            [2, "CIDEr", " - score"],
        ]:
            scores_static = [ret_orig_llavabench[metric_idx][metric_nam]]
            
            for compressed_scores in llavabench_static_metrics:
                scores_static.append(compressed_scores[metric_idx][metric_nam])
            
            _ = acc_pixel_red(
                scores_static[1:], scores_static[0], round_up=round_up, title=mod, y_axis=metric_nam + tit_score)
    
    if len(accs) > 0:
        # _ = acc_pixel_red(accs, orig_acc, round_up=round_up, title=mod)
        fig = acc_pixel_red(accs, orig_acc, round_up=round_up, title=mod, y_axis=y_axis)
        save_figure_as_pdf(fig, mod + "_" + dat)

def get_video_metadata(vid_pth):
    """
    Get the metadata of a given video file.

    Parameters
    ----------
    vid_pth : str
        Path to the video file.

    Returns
    -------
    num_frames : int
        Number of frames in the video.
    width : int
        Width of a frame in the video.
    height : int
        Height of a frame in the video.
    fps : float
        Frames per second of the video.
    """
    cap = cv2.VideoCapture(vid_pth)
    if cap.isOpened():
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    else:
        num_frames = 0
        width = 0
        height = 0
        fps = 0
    return num_frames, width, height, fps

def plot_simple_cdf(value_list, x_ax_tit="", names=None, integer_xticks=False, dataset_name=None, use_log_scale=False):
    """
    Plot one or multiple Cumulative Distribution Functions (CDFs).

    Parameters
    ----------
    value_list : list, array, or list of lists
        The list of values to plot the CDF for. If a list of lists is provided,
        multiple CDFs will be plotted.
    x_ax_tit : str, optional
        The title for the x-axis. Default is an empty string.
    names : list of str, optional
        Names/labels for the CDFs. Must be provided if value_list is a list of lists.
    integer_xticks : bool, optional
        Force integer ticks on the x-axis. Default is True.

    Returns
    -------
    int or list of int
        The number of elements in the input list(s).
    """

    figsize_mul = 1.15
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    ylabel_size = 29
    xlabel_size = 29

    # Detect single vs multiple datasets
    if isinstance(value_list[0], (int, float, np.number)):
        datasets = [value_list]
        labels = [names[0]] if names else [x_ax_tit or "CDF"]
    else:  # list of lists
        datasets = value_list
        if names is None or len(names) != len(datasets):
            raise ValueError("Must provide a `names` list of the same length as value_list.")
        labels = names

    # fig = plt.figure(figsize=[figsize_mul*6.4, figsize_mul*4.8], constrained_layout=True)
    # fig, ax = plt.subplots(figsize=(width, height + extra_space))
    fig, ax = plt.subplots(figsize=[figsize_mul*6.4, figsize_mul*4.8])

    counts = []
    for data, label in zip(datasets, labels):
        sorted_data = np.sort(data)
        n = len(sorted_data)
        y_cdf = np.arange(1, n + 1) / n
        plt.plot(sorted_data, y_cdf, marker='.', linestyle='-', markersize=4, label=label)
        counts.append(n)

    # plt.title('CDF of ' + dataset_name, fontsize=28)
    
    plt.xlabel(x_ax_tit, size=xlabel_size, x=0.39)
    # plt.ylabel('Probability (%)', size=ylabel_size)
    plt.ylabel(r'Probability ≤ x (%)', size=ylabel_size)

    if use_log_scale:
        plt.xscale('log')  # 🔑 This sets the x-axis to logarithmic scale

        from matplotlib.ticker import FixedLocator, FixedFormatter
        plt.tick_params(axis='x', which='major', labelsize=x_params_label_size)
        # plt.tick_params(axis='x', which='minor', labelsize=x_params_label_size) # Apply to minor ticks too

        # 1. Define the exact locations (values) where you want the ticks to appear
        tick_locations = [2e5, 4e5, 1e6]
        # 2. Define the corresponding labels you want to use for those locations
        # Using LaTeX for proper scientific notation is usually best
        tick_labels = [r'$2\times 10^5$', r'$4\times 10^5$', r'$10^6$']
        # OR if you prefer the standard format:
        # tick_labels = ['1e5', '2e5', '5e5', '1e6']

        # Get the current Axes object
        ax = plt.gca()
        # Set the custom tick locations
        ax.xaxis.set_major_locator(FixedLocator(tick_locations))
        # Set the custom tick labels
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))

    else:
        # desired_xticks = [1000, 2000, 3000]
        # desired_xlabels = ["1000", "2000", "3000"] # Optional: customize labels for readability
        # plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)
        # Calculate the start, end, and step for the ticks
        x_min = sorted_data.min()
        print(x_min)
        x_max = sorted_data.max()
        # Decide on the number of ticks you want, e.g., 5
        num_ticks = 3
        # Create the tick locations using np.linspace
        # This generates num_ticks evenly spaced values between x_min and x_max
        desired_xticks = np.linspace(x_min, x_max, num_ticks)
        # Optional: Format the labels to have no decimal places if desired
        desired_xlabels = [f"{int(x)}" for x in desired_xticks]
        # Now, apply these to your plot
        plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)

    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"], fontsize=y_params_label_size)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.055, 1.05)

    if len(datasets) > 1:
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        # plt.legend(loc='upper left', fontsize=20)
        # plt.legend(fontsize=20) # bbox_to_anchor=(0.5, -0.12), , frameon=False)
        # plt.legend(fontsize=16, loc='upper center', ncol=len(datasets)) # bbox_to_anchor=(0.5, -0.12), , frameon=False)
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),  # x=0.5 centers it, y=1.02 places it just above the plot
            bbox_transform=ax.transAxes,
            ncol=3,                      # number of columns if you have multiple legend items
            fontsize=16,
            frameon=False                # optional: removes the legend box
        )

    # # 🔑 Force integer ticks on x-axis if requested
    # if integer_xticks:
    #     ax = plt.gca()
    #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     ax.ticklabel_format(style='plain', axis='x')  # disable scientific notation

    if integer_xticks:
        x_params_label_size = 23.5
        desired_xticks = [3000000, 6500000, 10000000, 13000000]
        desired_xlabels = ["3mil", "6.5mil", "10mil", "13mil"] # Optional: customize labels for readability
        plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)

    plt.show()

    if len(counts) > 1:
        return counts, fig
    else:
        return counts[0], fig

def plot_frame_sizes(frame_sizes_list: list[tuple[int, int]], dataset_name=None):
    """
    Plots the frame sizes (width vs. height) for all video instances.

    Args:
        frame_sizes_list (list of tuple): A list where each tuple is (width, height)
                                           representing the dimensions of a video frame.
    """
    widths = [size[0] for size in frame_sizes_list]
    heights = [size[1] for size in frame_sizes_list]

    figsize_mul = 1.04
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    ylabel_size = 29
    xlabel_size = 29

    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)
    
    plt.scatter(widths, heights, alpha=0.6, s=20) # s is marker size
    
    plt.xticks(fontsize=x_params_label_size)
    plt.yticks(fontsize=y_params_label_size)

    x_min = min(widths)
    # x_min = 0
    x_max = max(widths)
    num_ticks = 4
    desired_xticks = np.linspace(x_min, x_max, num_ticks)
    desired_xlabels = [f"{int(x)}" for x in desired_xticks]
    plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)

    plt.xlabel('Width (pixels)', size=xlabel_size, x=0.39)
    plt.ylabel("Height (pixels)", size=ylabel_size)
    
    plt.title('Frame Sizes ' + dataset_name, fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Optional: Add axis limits based on data range for better visualization
    max_width = max(widths) if widths else 0
    max_height = max(heights) if heights else 0
    plt.xlim(0, max_width * 1.1)
    # plt.ylim(0, max_height * 1.1)

    plt.ylim(100, max_height + 50)

    # Add labels for common resolutions if desired (e.g., 640x480, 1280x720, 1920x1080)
    # This might clutter if there are many unique sizes, so consider carefully.
    common_res = [(640, 480), (1280, 720), (1920, 1080)]
    for w, h in common_res:
        if w <= max_width * 1.1 and h <= max_height * 1.1:
            plt.axvline(w, color='gray', linestyle=':', alpha=0.5, label=f'{w}x{h}')
            plt.axhline(h, color='gray', linestyle=':', alpha=0.5)
    # plt.tight_layout()
    plt.show()
    return fig

def print_top_n_frame_sizes(frame_sizes_list: list[tuple[int, int]], top_n: int = 5):
    """
    Identifies and prints the top_n most common video frame sizes
    along with their respective counts (instances).

    Args:
        frame_sizes_list (list of tuple): A list where each tuple is (width, height)
                                           representing the dimensions of a video frame.
        top_n (int): The number of top common frame sizes to display. Defaults to 5.
    """
    if not frame_sizes_list:
        print("No frame sizes to analyze.")
        return

    # Use Counter to count occurrences of each (width, height) tuple
    size_counts = Counter(frame_sizes_list)

    # Get the top_n most common sizes
    most_common_sizes = size_counts.most_common(top_n)

    print(f"\n--- Top {top_n} Most Common Frame Sizes ---")
    if not most_common_sizes:
        print("No common frame sizes found (list might be empty or all unique).")
        return

    # Calculate total instances for percentage calculation
    total_instances = sum(size_counts.values())

    for size, count in most_common_sizes:
        width, height = size
        percentage = (count / total_instances) * 100
        print(f"  {width}x{height}: {count} instances ({percentage:.2f}%)")
    print("-" * 40)

def evaluate_llm_scores(
    df: pd.DataFrame,
    ref_col: str,
    pred_col: str,
    score_type: Literal["rouge", "bleu", "cider"] = "rouge",
):
    """
    Evaluate text generation performance using ROUGE, BLEU, or CIDEr.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing reference and prediction columns.
    ref_col : str
        Column name for reference answers (ground truth).
    pred_col : str
        Column name for predictions from the LLM.
    score_type : {"rouge", "bleu", "cider"}
        The type of score to compute.

    Returns
    -------
    dict
        Dictionary of computed scores.
    """
    if isinstance(df, pd.DataFrame):
        refs = df[ref_col].astype(str).tolist()
        preds = df[pred_col].astype(str).tolist()
    else:
        refs = [df[ref_col]]
        preds = [df[pred_col]]

    if score_type == "rouge":
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=preds, references=refs)
        return results

    elif score_type == "bleu":
        bleu = evaluate.load("sacrebleu")
        results = bleu.compute(predictions=preds, references=[[r] for r in refs])
        return results

    elif score_type == "cider":
        # Format for CIDEr: list of {id: [ref]} and {id: [pred]}
        gts = {i: [ref] for i, ref in enumerate(refs)}
        res = {i: [pred] for i, pred in enumerate(preds)}
        cider_scorer = Cider()
        score, scores = cider_scorer.compute_score(gts, res)
        return {"CIDEr": score, "CIDEr_per_sample": scores}

    else:
        raise ValueError("score_type must be one of ['rouge', 'bleu', 'cider']")
