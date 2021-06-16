# Telemetry

Merged into main with [Pull Request 261](https://github.com/opsani/servox/pull/261).

Features:

* Allows the Servo and Connectors to gather and set arbitrary metadata key/value pairs
* Sends collected metadata to the OCO with the HELLO event request
* Gathers and sends the following metadata out of the box:
  * The servox version
  * The platform that servox is running on (as provided by [platform.platform()](https://docs.python.org/3/library/platform.html#platform.platform))
  * The namespace in which the servox pod is running (derived from the POD_NAMESPACE environment variable)
  * The version of kubernetes returned by [kubernetes.client.VersionApi.get_code()](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/VersionApi.md#get_code)
  * The platform of kubernetes returned by [kubernetes.client.VersionApi.get_code()](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/VersionApi.md#get_code)
  * The namespace specified in the kubernetes connector configuration

## Adding Metadata to Telemetry

Each assembled servo includes a telemetry attribute which is shared with its connectors.
This attribute functions much like a Dictionary mapping string keys to string values but
with a limited subset of operations: the index-of operator (`[]`) and a `remove` method
meant to remove metadata that may or may not be present within the telemetry object.

Telemetry can be added at any point after servo/connector instantiation, but it is recommended
to do so as part of the connector life cycle events per the following example from the kubernetes connector:

```
class KubernetesConnector(servo.BaseConnector):
    ...

    @servo.on_event()
    async def attach(self, servo_: servo.Servo) -> None:
        ...

        self.telemetry[f"{self.name}.namespace"] = self.config.namespace

        with self.logger.catch(level="DEBUG", message=f"Unable to set User Agent string for connector {self.name}"):
            async with kubernetes_asyncio.client.api_client.ApiClient() as api:
                v1 =kubernetes_asyncio.client.VersionApi(api)
                version_obj = await v1.get_code()
                self.telemetry[f"{self.name}.version"] = f"{version_obj.major}.{version_obj.minor}"
                self.telemetry[f"{self.name}.platform"] = version_obj.platform

    @servo.on_event()
    async def detach(self, servo_: servo.Servo) -> None:
        self.telemetry.remove(f"{self.name}.namespace")
        self.telemetry.remove(f"{self.name}.version")
        self.telemetry.remove(f"{self.name}.platform")
```

## Metadata Serialization

On HELLO requests to the Opsani OCO API, the telemetry backing dictionary is serialized and sent as json within the
`servo.api._post_event()` method

See the following for an example of the out of the box telemetry as serialized to a user-agent:

`{"event": "HELLO", "param": {"agent": "github.com/opsani/servox v0.10.4-alpha.0", "telemetry": {"servox.version": "0.10.4-alpha.0", "servox.platform": "Linux-5.8.0-55-generic-x86_64-with-glibc2.29", "kubernetes.namespace": "kubetest-test-telemetry-hello-1623872048", "kubernetes.version": "1.20", "kubernetes.platform": "linux/amd64"}}}`
