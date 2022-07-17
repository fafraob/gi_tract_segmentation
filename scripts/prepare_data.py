from typing import Iterable
import cv2
from glob import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


def process_df(df, image_paths: Iterable[str], data_dir: str):
    """
    Initial preparation of the dataset.
    """
    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(
        fault2)].reset_index(drop=True)

    df['case_str'] = df['id'].apply(lambda x: x.split('_')[0])
    df['day_str'] = df['id'].apply(lambda x: x.split('_')[1])
    df['slice_str'] = df['id'].apply(lambda x: '_'.join(x.split('_')[2:]))

    df['partial_path'] = (data_dir + '/train/' +
                          df['case_str'] + '/' +
                          df['case_str'] + '_' + df['day_str'] +
                          '/scans/' + df['slice_str'])

    path_df = pd.DataFrame({'partial_path': [x.rsplit(
        '_', 4)[0] for x in image_paths], 'image_path': image_paths})
    df = df.merge(path_df, on='partial_path').drop(columns=['partial_path'])

    df['slice_h'] = df['image_path'].apply(
        lambda x: int(x[:-4].rsplit('_', 4)[1]))
    df['slice_w'] = df['image_path'].apply(
        lambda x: int(x[:-4].rsplit('_', 4)[2]))

    df['px_spacing_h'] = df['image_path'].apply(
        lambda x: float(x[:-4].rsplit('_', 4)[3]))
    df['px_spacing_w'] = df['image_path'].apply(
        lambda x: float(x[:-4].rsplit('_', 4)[4]))

    stomach_df = df[df['class'] == 'stomach'][['id', 'segmentation']].rename(
        columns={'segmentation': 'st_seg_rle'})
    l_bowel_df = df[df['class'] == 'large_bowel'][['id', 'segmentation']].rename(
        columns={'segmentation': 'lb_seg_rle'})
    s_bowel_df = df[df['class'] == 'small_bowel'][['id', 'segmentation']].rename(
        columns={'segmentation': 'sb_seg_rle'})
    df = df.merge(stomach_df, on='id', how='left')
    df = df.merge(l_bowel_df, on='id', how='left')
    df = df.merge(s_bowel_df, on='id', how='left')
    df = df.drop_duplicates(subset=['id', ]).reset_index(drop=True)
    df['lb_seg_flag'] = df['lb_seg_rle'].apply(
        lambda x: not pd.isna(x)).astype(int)
    df['sb_seg_flag'] = df['sb_seg_rle'].apply(
        lambda x: not pd.isna(x)).astype(int)
    df['st_seg_flag'] = df['st_seg_rle'].apply(
        lambda x: not pd.isna(x)).astype(int)
    df['unique_seg_flag'] = df.agg(
        lambda x: f"{x['lb_seg_flag']}{x['sb_seg_flag']}{x['st_seg_flag']}", axis=1)
    df['n_segs'] = df['lb_seg_flag'] + df['sb_seg_flag'] + df['st_seg_flag']

    df.drop(columns=['class', 'segmentation'], inplace=True)

    return df


def add_dimensions(df: pd.DataFrame, pad_slices: int, stride: int):
    """
    Add slices before and after a given slice to create 3D-like context.
    """
    all_img_paths = set(df.image_path)

    def find_paths(path, pad_slices, stride, all_paths):
        image_paths = []
        base, file = path.rsplit('/', 1)
        slice_nr = int(file.split('_')[1])
        slice_idxs = np.arange(start=slice_nr+-pad_slices*stride,
                               stop=slice_nr+pad_slices*stride+1,
                               step=stride)
        for i in slice_idxs:
            file_parts = file.split('_')
            file_parts[1] = str(i).zfill(4)
            file_name = base + '/' + '_'.join(file_parts)
            if file_name in all_paths:
                image_paths.append(file_name)
            else:
                image_paths.append('')
        assert len(image_paths) == pad_slices * 2 + 1
        return image_paths

    df['image_paths'] = df['image_path'].progress_apply(
        lambda x: find_paths(x, pad_slices, stride, all_img_paths))

    return df


def decode_rle(mask_rle: str, shape: Iterable[int]):
    """
    Decode RLE into a binary segmentation mask.
    """
    s = np.array(mask_rle.split(' '), dtype=int)

    # Every even value is the start, every odd value is the 'run' length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    w, h = shape
    img = np.zeros((w * h), dtype=np.float32)

    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1

    return img.reshape(shape)


def mask_3D_from_rle(row: pd.Series, output_dir: str):
    """
    Creates a binary 3D segmentation mask from RLE segmentations, saves the mask as an image, and outputs the path to the mask.
    """

    slice_shape = (row.slice_w, row.slice_h)

    if not pd.isna(row.lb_seg_rle):
        lb_mask = decode_rle(row.lb_seg_rle, slice_shape)
    else:
        lb_mask = np.zeros(slice_shape, dtype=np.float32)

    if not pd.isna(row.sb_seg_rle):
        sb_mask = decode_rle(row.sb_seg_rle, slice_shape)
    else:
        sb_mask = np.zeros(slice_shape, dtype=np.float32)

    if not pd.isna(row.st_seg_rle):
        st_mask = decode_rle(row.st_seg_rle, slice_shape)
    else:
        st_mask = np.zeros(slice_shape, dtype=np.float32)

    mask_arr = np.stack([lb_mask, sb_mask, st_mask],
                        axis=-1).astype(np.float32)
    mask_path = os.path.join(output_dir, row['id'] + '_mask.png')
    cv2.imwrite(mask_path, mask_arr)

    return mask_path


def main():
    DATA_DIR = 'data'
    MASK_DIR = os.path.join(DATA_DIR, 'masks')
    if not os.path.isdir(MASK_DIR):
        os.makedirs(MASK_DIR)
        print('Created dir:', MASK_DIR)
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    image_paths = glob(os.path.join(DATA_DIR, 'train',
                                    '**', '*.png'), recursive=True)
    df = process_df(df, image_paths, DATA_DIR)
    df['mask_path'] = df.progress_apply(
        lambda row: mask_3D_from_rle(row, MASK_DIR), axis=1)
    df.drop(columns=['st_seg_rle', 'lb_seg_rle', 'sb_seg_rle'], inplace=True)
    df = add_dimensions(df, pad_slices=1, stride=2)
    df.to_csv(os.path.join(DATA_DIR, 'data.csv'), index=False)


if __name__ == '__main__':
    main()
