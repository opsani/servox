# Telemetry

Merged into main with [Pull Request 261](https://github.com/opsani/servox/pull/261).

Features:

* Allows the Servo and Connectors to gather and set arbitrary metadata key/value pairs
* Provides an api module method to serialize gathered metadata into a [user-agent](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent)
  string that will be sent on requests to the OCO API
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

## Metadata User Agent Serialization

On requests to the Opsani OCO API, the telemetry is serialized set as the user-agent request header
in the format defined within the [Mozilla User-Agent documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent)

During serialization into a user agent string, the keys of the telemetry object are treated as associations
between platforms, their details, and whatever arbitrary information is being collected. Keys with no `.`
seperator are treated as platform to version mappings. Keys containing `.` are treated as platform detail
mappings. Eg.

* `telemtry["kubernetes"] = 1.0` is serialized as `kubernetes/1.0`
* `telemtry["kubernetes.namespace"] ="some_namespace"` is serialized as `kubernetes (namespace some_namespace)`
(assuming a version has not been set for the `kubernetes` platform)

Note that the version of the platform can be specified either with the key of the platform, or
with the `<platform>.version` key

Note that the `servox` and `servox.*` keys are special case keys which will be serialized
with a platform of `github.com/opsani/servox`

See the following for an example of the out of the box telemetry as serialized to a user-agent:

`github.com/opsani/servox/0.10.3 (platform Linux-5.8.0-55-generic-x86_64-with-glibc2.29) kubernetes/1.20 (namespace kubetest-test-user-agent-1623359159; platform linux/amd64)`
