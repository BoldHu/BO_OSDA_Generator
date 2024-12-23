import maze
from maze.cif_download import download_cif
import os

def download_zeolite_files(zeolite_names, data_dir):
    """
    Download CIF files for specified zeolite names to the specified directory.

    Parameters:
    zeolite_names (list): List of zeolite names to download.
    data_dir (str): Directory where the CIF files will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    for zeolite in zeolite_names:
        # Check if the file already exists
        if os.path.exists(f"{data_dir}/{zeolite}.cif"):
            print(f"{zeolite} already exists in {data_dir}")
            continue
        try:
            download_cif(zeolite, data_dir=data_dir)
            print(f"Downloaded: {zeolite} to {data_dir}/{zeolite.lower()}.cif")
        except Exception as e:
            print(f"Error downloading {zeolite}: {e}")
        return zeolite_names

# Example usage
if __name__ == '__main__':
    zeolite_list = ["GOO", "ACO"]  # Add more zeolite names as needed
    download_zeolite_files(zeolite_list, "./simulation/zeolite_files")
