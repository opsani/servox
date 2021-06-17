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

    async def jsoncall(self, path, verb, in_obj, out_class, timeout=10):
        headers = {
            'content-type': 'application/json',
            'Authorization': 'Bearer {}'.format(self.auth_token),
            'Host': 'olas-backend',
        }
        async with aiohttp.ClientSession() as session:
            data = in_obj.json() if in_obj else ''
            async with getattr(session, verb)(path, data=data, headers=headers, timeout=timeout) as res:
                if res.status == 200:
                    obj = await res.json()
                    if out_class:
                        obj = parse_obj_as(out_class, obj)
                    return obj
                elif res.status == 204:
                    return data
                else:
                    err = await res.text()
                    servo.logger.info(f"jsoncall {path} error {res.status}, {err}")
                    return None

    async def upload_message(self, ts, msg):
        msg = sc.Message(ts=ts, msg=msg)
        return await self.jsoncall(f"{self.url}/{self.base}/messages", 'post', msg, sc.Message)

    async def upload_config(self, cfgdict):
        cfg = configuration.OLASConfiguration.parse_obj(cfgdict)
        return await self.jsoncall(f"{self.url}/{self.base}/config", 'put', cfg, '')

    async def predict(self, source):
        r = await self.jsoncall(f"{self.url}/{self.base}/prediction", 'get', '', sc.PredictionResult, timeout=30)
        if r is None:
            return None
        # convert response, a Pydantic model, to a dictionary
        r = r.dict()
        if err := r[source]['error']:
            servo.logger.info(f"Prediction {self.app_id} failed with error: {err}")
            return None
        return r[source]['value']

    async def upload_metrics(self, metrics):
        m = sc.Metrics.parse_obj(metrics)
        return await self.jsoncall(f"{self.url}/{self.base}/metrics", 'post', m, sc.Id)

    async def get_pod_model(self):
        return await self.jsoncall(f"{self.url}/{self.base}/model", 'get', '', sc.PodModelWithId)
