import aiohttp
from pydantic import parse_obj_as

import servo
from servo.connectors.olas import configuration
from servo.connectors.olas import server_classes as sc


class OLASClient:
    def __init__(self, url, account, app_id, auth_token):
        self.url = url
        self.account = account
        self.app_id = app_id
        self.base = f"accounts/{account}/applications/{app_id}/olas"
        self.auth_token = auth_token

    async def jsoncall(self, path, in_obj, out_class, timeout=10):
        headers = {
            'content-type': 'application/json',
            'Authorization': 'Bearer {}'.format(self.auth_token),
            'Host': 'olas-backend',
        }
        async with aiohttp.ClientSession() as session:
            data = in_obj.json() if in_obj else ''
            async with session.put(path, data=data, headers=headers, timeout=timeout) as res:
                if res.status == 200:
                    obj = await res.json()
                    if out_class:
                        obj = parse_obj_as(out_class, obj)
                    return obj
        servo.logger.info(f"jsoncall {path} error")
        return None

    async def upload_message(self, ts, msg):
        msg = sc.Message(ts=float(ts), msg=msg)
        return await self.jsoncall(f"{self.url}/{self.base}/upload_message", msg, sc.Id)

    async def upload_config(self, cfgdict):
        cfg = configuration.OLASConfiguration.parse_obj(cfgdict)
        return await self.jsoncall(f"{self.url}/{self.base}/upload_config", cfg, sc.Id)

    async def predict(self, source):
        traffic = sc.Prediction(src=source)
        r = await self.jsoncall(f"{self.url}/{self.base}/predict", traffic, sc.PredictionResult, timeout=30)
        if r is None:
            return None
        if r.err:
            servo.logger.info(f"Prediction {self.app_id} failed with error: {r.err}")
            return None
        return r.value

    async def upload_metrics(self, metrics):
        m = sc.Metrics.parse_obj(metrics)
        return await self.jsoncall(f"{self.url}/{self.base}/upload_metrics", m, sc.Id)

    async def get_pod_model(self):
        return await self.jsoncall(f"{self.url}/{self.base}/get_pod_model", '', sc.PodModelWithId)
