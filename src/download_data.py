import os
import requests
import zipfile
from tqdm import tqdm
import sys
sys.path.append('../src')
from ais_to_parquet_group_by_type import fn

# --- Folder setup ---

PROCESSED_DIR = '../data/processed'
RAW_DATA_DIR = '../data/raw'

def download_and_extract_ais_data(base_url, season, file_names):
    # one week of AIS data files
    urls = [base_url + fname for fname in file_names]
    target_dir  = RAW_DATA_DIR + f'/{season}'
    # make sure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    target_paths = [os.path.join(target_dir, fname) for fname in file_names]

    # --- Streamed download with progress bar ---
    for url, zip_path in zip(urls, target_paths):
        print(f'⬇️ Downloading {url} ...')
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Total size in bytes.
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)



        with open(zip_path, 'wb') as f:
            for data in response.iter_content(block_size):
                tqdm_bar.update(len(data))
                f.write(data)
        tqdm_bar.close()

        if total_size != 0 and tqdm_bar.n != total_size:
            print('⚠️ Download may be incomplete.')
        else:
            print(f'✅ ZIP file downloaded to {zip_path}')

        # --- Extract ZIP ---
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f'📦 Extracted files to {target_dir}')

def convert_ais_to_parquet_by_type(season):
    input_dir = RAW_DATA_DIR + f'/{season}'
    output_dir = PROCESSED_DIR + f'/{season}'
        # --- Find and load CSV(s) ---
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    csv_paths = [os.path.join(input_dir, f) for f in csv_files]

    # Choose input and output paths
    for csv_file in csv_files:
        print(f'Converting {csv_file} to Parquet...')
        input_csv = os.path.join(input_dir, csv_file)
        output_dir = '../data/processed/' + season + '/' + csv_file.replace('.csv', '')
        print(f'Processing {input_csv} ...')
        fn(input_csv, output_dir)
        print(f'✅ Conversion complete. Parquet data stored in {output_dir}')