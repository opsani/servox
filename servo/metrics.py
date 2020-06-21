# ---------------------------


class Metric:
    # name, unit
    pass


# Models a collection of metrics
class Metrics:
    pass


# Models a su
class ScalarMetric:
    pass


# Models a time-series metrics
class TimeSeriesMetric:
    pass


class Descriptor:
    pass


class Component:
    pass


# {"application": {"components": {"web": {"settings": {"cpu": {"value": 0.25, "min": 0.1, "max": 1.8, "step": 0.1, "type": "range"}, "replicas": {"value": 1, "min": 1, "max": 2, "step": 1, "type": "range"}}}}}, "measurement": {"metrics": {"requests_total": {"unit": "count"}, "throughput": {"unit": "rpm"}, "error_rate": {"unit": "percent"}, "latency_total": {"unit": "milliseconds"}, "latency_mean": {"unit": "milliseconds"}, "latency_50th": {"unit": "milliseconds"}, "latency_90th": {"unit": "milliseconds"}, "latency_95th": {"unit": "milliseconds"}, "latency_99th": {"unit": "milliseconds"}, "latency_max": {"unit": "milliseconds"}, "latency_min": {"unit": "milliseconds"}}}}
class Setting:
    # name, step, type, value, min, max
    pass
