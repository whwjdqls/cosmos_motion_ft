"""Convert saved Nymeria timecodes using the original headset VRS."""
import json
import sys

from projectaria_tools.core import data_provider


if __name__ == '__main__':
    query = json.load(sys.stdin)
    provider = data_provider.create_vrs_data_provider(sys.argv[1])
    if provider is None:
        raise RuntimeError('Head motion VRS could not be opened')
    result = [int(provider.convert_from_timecode_to_device_time_ns(int(t)))
              for t in query['timecode_ns']]
    print('SCENE_CLOCK_JSON=' + json.dumps({'device_ns': result}), flush=True)
