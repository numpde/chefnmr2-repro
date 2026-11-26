"""Acquire Chef's exact ANTLR source archive through its existing fetch owner."""

import json
import logging
from pathlib import Path
import sys

from fetch_sdk import fetch


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    (pool, declaration) = map(Path, sys.argv[1:])
    source = json.loads(declaration.read_text())
    (release, filename) = source['url'].rsplit('/', 1)

    if filename != source['name']:
        raise ValueError(f'Cannot acquire ANTLR: URL {filename=} differs from the declared {source["name"]=}')

    fetch(pool, release + '/', filename, source['sha256'])
