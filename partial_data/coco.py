import json
import os

import pandas as pd


def load_coco_annotations_as_dataframe(annotations_filepath, local_image_dir):
    # Load raw data from disk
    with open(annotations_filepath, 'r') as fp:
        instances = json.load(fp)

    # Load annotation info into dataframe
    df_ann = (
        pd.DataFrame(instances['annotations'])
        .drop(columns=['segmentation', 'area'])
        .rename(columns={'id': 'instance_id'})
    )

    # Load category info into dataframe
    df_cat = (
        pd.DataFrame(instances['categories'])
        .drop(columns=['supercategory'])
        .rename(columns={'id': 'category_id', 'name': 'category_name'})
    )
    df_cat['category_name_orig'] = df_cat['category_name']
    df_cat['category_name'] = df_cat['category_name'].str.lower().str.replace(' ', '_')

    # Load image info
    df_image = pd.DataFrame(instances['images']).rename(columns={'id': 'image_id'})
    df_image['image_filepath'] = df_image['file_name'].apply(lambda x: os.path.join(local_image_dir, x))
    df_image.drop(columns=['license', 'file_name', 'date_captured', 'flickr_url'], inplace=True)

    # Combine all info into single dataframe
    df_comb = pd.merge(df_ann, df_cat, on='category_id', how='left')
    df_comb = pd.merge(df_comb, df_image, on='image_id', how='left')

    # Convert bounding boxes to normalized coordinates
    # Original box coordinates are [x,y,width,height], measured from the top left image corner, and 0-indexed
    df_comb['wmin'] = df_comb['bbox'].str[0] / df_comb['width']
    df_comb['hmin'] = df_comb['bbox'].str[1] / df_comb['height']
    df_comb['wmax'] = (df_comb['bbox'].str[0] + df_comb['bbox'].str[2]) / df_comb['width']
    df_comb['hmax'] = (df_comb['bbox'].str[1] + df_comb['bbox'].str[3]) / df_comb['height']

    return df_comb
