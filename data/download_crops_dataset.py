import os
import argparse
from minio import Minio
from tqdm import tqdm

# AirLab server config
BUCKET = "sort3d"
ENDPOINT = "airlab-share-01.andrew.cmu.edu:9000"

# Public keys (for downloading)
ACCESS_KEY = "lNJg8FovonIvAuomH7xH"
SECRET_KEY = "JOdaE6OfENXRtMs8U4NdRo2vjrNsgJ0eV1D2zF4E"


def get_from_server(client: Minio, bucket_name, source_name, target_name):
    """
    Downloads a specific file from server using Minio

    Args: 
    client: Minio client object with set up with keys
    bucket_name: str name of data bucket
    source_name: name of file on server
    target_name: name of file locally

    Returns: True
    """
    print(f"Downloading {source_name} from {bucket_name}...")
    client.fget_object(bucket_name, source_name, target_name)
    print(f"Successfully downloaded {source_name} to {target_name}!")

    return True


def download(args):
    """
    Configures download client and loops through files
    """
    client = Minio(ENDPOINT,
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                secure=True)
    
    if not os.path.exists(args.download_path):
        os.mkdir(args.download_path)

    file = 'captions.zip'
    file_path = 'captions_dataset/captions.zip'
    target_name = os.path.join(args.download_path, file)
    res = get_from_server(client,
                        BUCKET,
                        source_name=file_path,
                        target_name=target_name)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_path', default='data', help="Directory to store downloaded data.")

    args = parser.parse_args()

    download(args)