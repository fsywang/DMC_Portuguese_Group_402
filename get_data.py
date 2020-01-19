import requests 
import os
import argparse
import zipfile

parser = argparse.ArgumentParser()

def folder(save_dir):
    # Create target Directory if don't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"Directory {save_dir} Created ")
    
    return save_dir


def download_and_process_data(url, save_dir):
    # Check if the folder exisits, if not create

    filename = url.split('/')[-1]

    r = requests.get(url) # create HTTP response object 

    # Download the file
    print('save_dir: ', save_dir)
    print('filename: ', filename)
    filepath = os.path.join(save_dir, filename)
    print("Downloading the content")
    with open(filepath,'wb') as f: 
        f.write(r.content) 

    # Unzip the file if needed:
    if zipfile.is_zipfile(filepath):
        print("Unzipping the downloaded file")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(save_dir)


if __name__ == '__main__':
    parser.add_argument('--url', type=str,
            required=False, help='The URL from where the data can be download',
            default='https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip')

    parser.add_argument('--save_dir', type=folder,
            required=False, help='The folder where the data will be saved',
            default='./data')

    args = parser.parse_args()

    download_and_process_data(args.url, args.save_dir)
