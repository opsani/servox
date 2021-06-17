from datetime import datetime, timezone

TIME_STAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


def epoch_to_str(ts: float) -> str:
    '''
    Convert epoch time to UTC datetime object then to human readbale string.

    A typical input to this function is obtained through time.time().

    For ISO format, see https://docs.python.org/3/library/datetime.html#datetime.datetime.isoformat.

    Besides, 'Z' is suffixed in order to fully comply with ISO 8601.
    '''
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec='milliseconds')[:-6] + 'Z'


def str_to_epoch(ts: str) -> float:
    '''
    Convert ISO formatted time string to UTC datetime object then to epoch time
    '''
    return datetime.fromisoformat(ts[:-1] + '+00:00').timestamp()
