"""Run the declared checkpoint downloader through the shared acquisition owner."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--proxy', required=True)
    args = parser.parse_args()
    docker = ['docker', '--host', args.host]
    composition = json.loads(subprocess.check_output(docker + [
        'compose', '--project-directory', str(ROOT), '--env-file', '/dev/null',
        '-f', str(ROOT / 'compose/weights.yaml'), 'config', '--format', 'json'], text=True))
    download = composition['services']['download']
    # The application owns its image, command, resource limits and volume access.
    # The acquisition owner supplies networking and helper lifecycle.
    del download['network_mode']
    del download['depends_on']
    volume = composition['volumes']['checkpoints']
    caller = {'services': {'worker-online': download},
              'volumes': {'checkpoints': {'external': True, 'name': volume['name']}}}
    bake = json.loads(subprocess.check_output(docker + [
        'buildx', 'bake', '-f', str(ROOT / 'docker-bake.hcl'), '--print', 'provider'], text=True))
    image = bake['target']['provider']['args']['PYTHON_BASE']
    with tempfile.TemporaryDirectory(prefix='chefnmr2-weights-') as directory:
        overlay = Path(directory) / 'worker.json'
        overlay.write_text(json.dumps(caller))
        subprocess.run(['bash', str(ROOT / 'make/acquisition.sh'), args.host, image, args.proxy,
                        '--compose', str(overlay), 'worker-online', 'chefnmr2-weights-writer'], check=True)


if __name__ == '__main__':
    main()
