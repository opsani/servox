import asyncio
import math
import traceback
from typing import List, Dict

import aiohttp
from pydantic import BaseModel

import servo


class PrometheusData(BaseModel):
    result: List
    resultType: str


class PrometheusReply(BaseModel):
    data: PrometheusData
    status: str


class PrometheusVector(BaseModel):
    metric: Dict
    value: List


class Histogram:
    def __init__(self, buckets):
        # buckets = [ (0.5, counter) , (1.0, count) ...]  item in (le, count) in sorted order
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
            left = float(buckets[low - 1][0])
            low_v = buckets[low - 1][1]

        high_v = buckets[low][1]
        if low_v == high_v:
            return left

        right = float(buckets[low][0])
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


class PrometheusQuery:
    def __init__(self, url):
        self.url = url

    async def querys(self, querys, timeout=10):
        async def fetch(url, params={}, timeout=timeout):
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    # servo.logger.info(f'PROM response {response}')
                    if response.status == 200:
                        obj = await response.json()
                        return obj

                    servo.logger.error(f'Promtheus status code "{response.status}"')
            return {}

        tasks = [asyncio.create_task(fetch(f'{self.url}/api/v1/{method}', params))
                 for method, params in querys]
        responses = await asyncio.gather(*tasks)
        return responses

    def extract_vector_list(self, res):
        return [self.extract_vector(r) for r in res]

    def extract_vector(self, r):
        try:
            reply = PrometheusReply.parse_obj(r)
            result = reply.data.result
            resultType = reply.data.resultType
            if resultType == 'vector':
                if not result:
                    return 0
                metric = PrometheusVector.parse_obj(reply.data.result[0])
                return float(metric.value[-1])
            elif resultType == 'matrix':
                # More than one vector value. Leave to the caller to process it.
                return result
            else:
                servo.logger.error(f'Unhandled promtheus result type "{resultType}"')
        except Exception:
            servo.logger.error(traceback.format_exc())
            return math.nan

    def merge_buckets(self, metrics):
        merge = {}  # dict( {'le': value} = value)
        for m in metrics:
            label = m['metric']
            le = float(label['le'])
            values = m['values']
            first = values[0]
            last = values[-1]
            value = int(last[1]) - int(first[1])
            # duration = last[0] - first[0]
            # print(podname, le, duration, value)
            merge[le] = merge.get(le, 0.0) + value
        buckets = sorted(merge.items())
        return buckets

    def merge_rate(self, matrix):
        total_rate = 0.0
        for m in matrix:
            values = m['values']
            first = values[0]
            last = values[-1]
            value = int(last[1]) - int(first[1])
            duration = last[0] - first[0]
            rate = value / duration if duration else 0
            total_rate += rate
            # print(podname, le, duration, value)
        return total_rate
