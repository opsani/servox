import asyncio

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

import servo


class Histogram:
    def __init__(self, buckets):
        self.buckets = buckets

    def get_pn(self, pn):
        buckets = self.buckets
        low = 0
        high = len(buckets)
        if not high:
            return 0  # XXX: 0 means N/A, None

        x = pn / 100.0 * buckets[-1][1]
        # print(x, buckets[-1][1])
        while low < high:
            mid = (low + high) // 2
            if buckets[mid][1] < x:
                # print("mid < x", buckets[mid][1], x, "low", low, ">", mid +1)
                low = mid + 1
            else:
                # print("mid >= x", buckets[mid][1], x, "high", high, ">", mid)
                high = mid
        if low == 0:
            # the histogram has implicit (0,0) starting point
            left = 0
            low_v = 0
        else:
            left = float(buckets[low - 1][0]['le'])
            low_v = buckets[low - 1][1]

        high_v = buckets[low][1]
        if low_v == high_v:
            return left

        right = float(buckets[low][0]['le'])
        # print (buckets)
        # print (low_v, x, high_v)
        # print (left, right - left, x, x -low, high_v, low_v, high_v - low_v)
        return left + (right - left) * (x - low_v) / (high_v - low_v)

    def __str__(self):
        buckets = [int(value) for label, value in self.buckets]
        buckets = [n - c for c, n in zip(buckets[:-1], buckets[1:])]
        while buckets and not buckets[-1]:
            del buckets[-1]
        dist = "\t".join(map(str, buckets))
        return f'{sum(buckets)/60:.1f}\t[{dist}]'


class EnvoyStats:
    def __init__(self, body):
        metrics = {}
        ms = text_string_to_metric_families(body)
        for family in ms:
            for sample in family.samples:
                # if sample.name.endswith('upstream_rq_time_bucket'):
                #    print (family.name, sample.name, sample.labels, sample.value)
                metrics.setdefault(sample.name, []).append((sample.labels, sample.value))
        self.metrics = metrics

    def get_buckets(self, name, label={}):
        buckets = self.metrics.get(name, [])
        buckets = [b for b in buckets if label.items() <= b[0].items()]
        return buckets

    def get_delta_bucket(self, name, label={}, prev=[]):
        total = self.get_buckets(name, label)
        # servo.logger.info(f'name {name} label {label}\ntotal_bucket {pprint.pformat(total)}\nprev {pprint.pformat(prev)}')
        if not prev:
            return total, total
        return self.delta_bucket(prev, total), total

    def get_histogram(self, name, label={}, prev=[]):
        delta, total = self.get_delta_bucket(name, label, prev)
        return Histogram(delta), total

    def delta_bucket(self, old, new):
        assert len(old) == len(new)
        delta = [(n[0], n[1] - o[1]) for o, n in zip(old, new)]
        return delta

    def merge_bucket(self, ba, bb):
        assert len(ba) == len(bb)
        return [(a[0], a[1] + b[1]) for a, b in zip(ba, bb)]

    def get_sample(self, name, label):
        assert isinstance(label, dict)
        m = self.metrics.get(name, [])
        for ll, v in m:
            if ll == label:
                return v

    def get_rate(self, name, label, interval, prev=0):
        v = self.get_sample(name, label)
        return (v - prev) / interval if interval else 0, v


async def get_envoy_stats(url, session):
    '''return EnvoyStats object'''
    async with session.get(url, timeout=1) as res:
        if res.status == 200:
            return EnvoyStats(res.text), ''
    servo.logger.error("Error envoy stats fetch error %s" % (url))
    return None, "Fetch failed"


async def get_envoy_stats_nodes(baseurls):
    '''
    return a list of metrics from list of urls.
    '''
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(get_envoy_stats(url, session)) for url in baseurls]
        return await asyncio.gather(*tasks)
