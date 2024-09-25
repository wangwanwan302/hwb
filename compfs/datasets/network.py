"""
Utilities and helpers for retrieving the datasets
"""
# stdlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


def download_http_if_needed(path: Path, url: str) -> None:
    """
    Helper for downloading a file, if it is now already on the disk.

    Parameters
    ----------
    path: Path
        Where to download the file.
    url: URL string
        HTTP URL for the dataset.
    """
    path = Path(path)

    if path.exists():
        return

    if url.lower().startswith("http"):
        urllib.request.urlretrieve(url, path)  # nosec
        return

    raise ValueError(f"Invalid url provided {url}")


def unarchive_if_needed(path: Path, output_folder: Path) -> None:
    """
    Helper for uncompressing archives. Supports .tar.gz and .tar.

    Parameters
    ----------
    path: Path
        Source archive.
    output_folder: Path
        Where to unarchive.
    """
    if str(path).endswith(".tar.gz"):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(path=output_folder) # nosec
        tar.close()
    elif str(path).endswith(".tar"):
        tar = tarfile.open(path, "r:")
        tar.extractall(path=output_folder) # nosec
        tar.close()
    elif str(path).endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(output_folder) # nosec
    else:
        raise NotImplementedError(f"archive not supported {path}")


def download_if_needed(
    download_path: Path,
    http_url: str,  # used for downloading from a HTTP URL
    unarchive: bool = False,  # unzip a downloaded archive
    unarchive_folder: Optional[Path] = None,  # unzip folder
) -> None:
    """
    Helper for retrieving online datasets.

    Parameters
    ----------
    download_path: str
        Where to download the archive
    http_url: str
        Set this if you want to download from a HTTP URL
    unarchive: bool
        Set this if you want to try to unarchive the downloaded file
    unarchive_folder: str
        Mandatory if you set unarchive to True.
    """
    download_path = Path(download_path)
    if http_url is not None:
        download_http_if_needed(download_path, http_url)
    else:
        raise ValueError("Please provide a download URL")

    if unarchive and unarchive_folder is None:
        raise ValueError("Please provide a folder for the archive")
    if unarchive and unarchive_folder is not None:
        try:
            unarchive_if_needed(download_path, unarchive_folder)
        except BaseException as e:
            print(f"Failed to unpack {download_path}. Error {e}")
            download_path.unlink()
