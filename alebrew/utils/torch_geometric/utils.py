"""
Copyright (c) 2023 PyG Team <team@pyg.org>

Trimmed-down pytorch_geometric:
"ALEBREW: The Atomic Learning Environment for Building REliable interatomic neural netWork potentials" 
uses pytorch_geometric (https://github.com/pyg-team/pytorch_geometric, https://arxiv.org/abs/1903.02428) 
framework for the most basic graph data structures.

We follow the same approach as NequIP (https://github.com/mir-group/nequip) and MACE (https://github.com/ACEsuit/mace) 
and copy their code here. This approach avoids adding many unnecessary second-degree dependencies and simplifies installation. 
Only a small subset of pytorch_geometric is included and modified, as necessary for our code.
"""
import os
import os.path as osp
import ssl
import urllib
import zipfile


def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path

    if log:
        print("Downloading", url)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)