"""Acquire the independently pinned tool through Chef's artifact verifier."""

import json
import logging
from pathlib import Path
import sys

from fetch_sdk import fetch


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    (pool, declaration) = map(Path, sys.argv[1:])
    pin = json.loads(declaration.read_text())

    fetch(pool, '', pin['name'], pin['sha256'], url=pin['url'])
