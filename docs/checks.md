# Checks

Merged into main with [Pull Request 29](https://github.com/opsani/servox/pull/29).

Features:

* Create check from method
* Create multiple checks in a class
* Create checks from an iterable (e.g. a list)
* Required checks
* Check metadata
* Filtering checks
* Halting runs
* Instrumentation
* Enhanced CLI
* Checkable Protocol
* Safe and productive by default

## Creating a Check

Checks can now be created with a decorator:

```python
import servo


@servo.check("Something...")
def check_something() -> None:
    ...
```

The decorator transforms the function into a method that returns a `Check`
object. When called it, it invokes the original method implementation and
determines success/failure based on the return type:

```python
@servo.check("Success")
def check_success() -> bool:
    return True # the check succeeded

@servo.check("Failure")
def check_failure() -> bool:
    return False # failed

@servo.check("Fail by exception")
def check_exception() -> None:
    raise RuntimeError("Something went wrong")
```

Exceptions are guarded for you. Write the shortest code possible that can check
the condition.

You can also return a message that will be displayed in the CLI (more on this
later):

```python
@servo.check("Message")
def check_message() -> str:
    return "Success message"
```

or return a bool and message to do both at once:

```python
@servo.check("Tuple outcome")
def check_tuple() -> Tuple[bool, str]:
    return (True, "Returning a tuple value works fine")
```

A check that returns `None` and doesn't raise is a success:

```python
@servo.check("None")
def check_none() -> None:
    print("Whatever I do here is a success unless I raise an error.")
```

### Check metadata

Checks can be enriched with metadata:

```python
@servo.check("Metadata",
    description="Include a longer detailed description here...",
    id="metadata",
    tags=("fast", "low_priority"),
)
def check_metadata() -> None:
    ...
```

Metadata comes into play a bit later. But for now, keep in mind that the `id` is
a short unique identifier that will be auto-assigned if unspecified and `tags`
is a set of lightweight descriptors about check behavior and context.

### Creating Checks

Checking several conditions in one connector can be verbose even with the
decorator:

```python
class SomeConnector(BaseConnector):
    @event()
    async def check(self) -> List[Check]:
        @check("one")
        def check_one() -> None:
            ...

        @check("two")
        def check_two() -> None:
            ...

        @check("three")
        def check_three() -> None:
            ...

        return [check_one(), check_two(), check_three()]
```

but more importantly, there is no way to work with the collection. It's an all
or nothing operation where all the checks are run and returned every time you
call `servo check`.

We can do better on both fronts:

```python
class SomeChecks(BaseChecks):
    @check("one")
    async def check_one(self) -> None:
        ...

    @check("two")
    async def check_two(self) -> None:
        ...

    @check("three")
    async def check_three(self) -> None:
        ...

class SomeConnector(BaseConnector):
    @event()
    async def check(self) -> List[Check]:
        return await SomeChecks.run(self.config)
```

The checks are now encapsulated into a standalone class that can be tested in
isolation. The check event handler is now nice and tidy.

The `BaseChecks` class has some interesting capabilities. It enforces a policy
that all instance methods are prefixed with `check_` and return a `Check` object
or are designated as helper methods by starting with an underscore:

```python
class ValidExampleChecks(BaseChecks):
    @check("valid")
    async def check_valid(self) -> None:
        ...

    def _reverse_string(self, input: str) -> str:
        return input.reverse()

class InvalidExampleChecks(BaseChecks):
    # Forgot to decorate -- wrong return value
    async def check_valid(self) -> None:
        ...

    def not_a_check(self) -> None:
        ...

    @check("not checkable return value")
    async def check_invalid_return(self) -> int:
        # Cannot be coerced into a check result
        123
```

Checks are always executed in *method definition order* (or top-to-bottom if you
prefer). This becomes important in a second.

### Required checks

Not all checks are created equal. There are some checks that upon failure imply
that all following checks will already have *implicitly failed*. Such checks can
be described as **required**.

Consider the example of implementing checks for Kubernetes. The very first thing
that it makes sense to do is check if you can connect to the API server (or run
`kubectl` in a subprocess). If this check fails, then it makes zero sense to
even attempt to check if you can create a Pod, read Deployments, have the
required secrets, etc.

To handle these cases, we can combine the notion of a required check with the
guarantee of checks executing in method definition order to express these
relationships between checks:

```python
class KubernetesChecks(BaseChecks):
    @check("API connectivity", required=True)
    def check_api(self) -> None:
        raise RuntimeError("can't reach API")

    @check("read namespace")
    def check_read_namespace(self) -> None:
        ...

    @check("has running Pods")
    def check_pods(self) -> None:
        ...

    @check("read deployments", required=True)
    def check_read_deployments(self) -> None:
        raise RuntimeError("can't read Deployments")

    @check("containers have resource limits")
    def check_resource_limits(self) -> None:
        ...
```

In this example, we have two required checks that act as circuit breakers to
halt execution upon failure. If `check_api` fails, then no other checks will be
run (more on this in a minute) and you will get a single error to debug. If
`check_api` succeeds but `check_read_deployments` fails, then the resource
limits won't be checked because if you can't see the Deployment you can't get it
to its containers and the requests/limits values.

### New event handler

The check metadata mentioned earlier combines with required checks and the
execution order guarantee to provide some very nice capabilities for controlling
check execution.

To support these enhancements, the method signature of the `check` event handler
has changed:

```python
class NewEventConnector(BaseConnector):
    @on_event()
    async def check(self,
        matching: Optional[Filter] = None,
        halt_on: HaltOnFailed = HaltOnFailed.requirement
    ) -> List[Check]:
        ...
```

There are a few things going on here. We have two new positional parameters:
`matching` and `halt_on`. Let's look at these one at a time.

### Filtering checks

The `matching` argument is an instance of `servo.checks.Filter` which looks like
this (edited down for brevity and clarity):

```python
class Filter(BaseModel):
    name: Union[None, str, Sequence[str], Pattern[str]] = None
    id: Union[None, str, Sequence[str], Pattern[str]] = None
    tags: Optional[Set[str]] = None

    def matches(self, check: Check) -> bool:
        ...
```

These are the same attributes discussed earlier in the check metadata section.
The filter matches against checks using AND semantics (all constraints must be
satisfied for a match to occur).

The `name` and `id` attributes can be compared against an exact value (type
`str`),  a set of possible values (type `Sequence[str]`, which includes lists,
sets, and tuples of strings), or evaluated against a regular expression pattern
(type `Pattern[str]`).  Values are compared case-sensitively. `id` values are
always lowercase alphanumeric characters or `_`.

Tags are evaluated with set intersection semantics (the constraint is satisfied
if the check has any tags in common with the filter).

A value of `None` always evaluates positively for the particular constraint.

### Halting check execution

Depending on what you are doing, it can be desirable to handle failing checks
differently. You may wish to fail fast to identify a blocked requirement or you
may want to run every check and get a sense for how broken your setup is all in.

This is where the `halt_on` parameter comes in. `halt_on` is a value of the
`HaltOnFailed`  enumeration which looks like:

```python
class HaltOnFailed(enum.StrEnum):
    """HaltOnFailed is an enumeration that describes how to handle check failures.
    """

    requirement = "requirement"
    """Halt running when a required check has failed.
    """

    check = "check"
    """Halt running when any check has failed.
    """

    never = "never"
    """Never halt running regardless of check failures.
    """
```

Selecting the appropriate `halt_on` value lets you decide how much feedback you
want to gather in a given check run.

### Configuration

All of the above changes are pretty hard to utilize without an interface. As
such, configuration for checks can be done in the checks section of the `servo.yaml`,
as defined by the [ChecksConfiguration](../servo/configuration.py#L468) class.
The checks configuration is not required explicitly, and if not specified will
run with default options. Below is an example of the checks configuration with
all configurable options specified explicitly.

```servo.yaml
    opsani_dev:
      ...
    checks:
      connectors: ['opsani-dev']
      name: ['Connectivity to Kubernetes']
      id: ['check_kubernetes_connectivity']
      quiet: False
      verbose: False
      progressive: False
      wait: 30m
      delay: 10s
      halt_on: critical
      remedy: True
      check_halting: False
```

By default, checks and any associated remedies run asynchronously, but remedies
can be applied sequentially upon check failure by setting `check_halting` to True

```console
    checks:
      check_halting: True
```

Results from checks can be output into a table
```console
    checks:
      progressive: False
```

```console
CONNECTOR                        STATUS    ERRORS
test.optimizer.com/test          X FAILED  (1/1) Opsani API connectivity: ['Response status code: 404']
opsani_dev                       √ PASSED
opsani-dev:kubernetes            √ PASSED
opsani-dev:prometheus            X FAILED  (1/1) Connect to "http://localhost:9090": ['caught exception (ConnectError): [Errno 61] Connection refused']
opsani-dev:kube-metrics          √ PASSED
```

We can run a check by name:

```console
    checks:
      name: ['Connectivity to Kubernetes']
```

Or a set of IDs comma separated:

```console
    checks:
      id: ['check_kubernetes_connectivity']
```

Or every check that contains "exec" (strings in slashes "/like this/" are
compiled as regex):

```console
    checks:
      name: ["/.*exec.+/"]
```

And set the halting behavior in the face of failures:

```console
    checks:
      halt_on: common
```

### Creating Checks from an Iterable

Sometimes you have a collection of homogenous items that need to be checked. A
common example is a list of queries for a metrics provider like Prometheus:

```yaml
prometheus:
  base_url: http://localhost:9091/
  metrics:
  - name: throughput
    query: rate(http_requests_total[1s])[3m]
    unit: rps
  - name: error_rate
    query: rate(errors)
    unit: '%'
  - name: go_threads
    query: gc_info
    unit: threads
  - name: go_memstats_alloc_bytes
    query: go_memstats_alloc_bytes
    unit: bytes
```

We don't want to handwrite a method for each of these and if we just loop over
it, we can't use filters to focus on the failure cases -- making debugging
slower and noisier.

What we want is the ability to synthesize a checks class without having to write
the code by hand:

```python
from typing import Optional
import servo


class PrometheusConnector(servo.BaseConnector):
    config: PrometheusConfiguration

    @servo.on_event()
    async def check(self,
        matching: Optional[servo.Filter] = None,
        halt_on: servo.HaltOnFailed = servo.HaltOnFailed.requirement
    ) -> List[Check]:
        start, end = datetime.now() - timedelta(minutes=10), datetime.now()
        async def check_query(metric: PrometheusMetric) -> str:
            result = await self._query_prom(metric, start, end)
            return f"returned {len(result)} TimeSeries readings"

        # wrap all queries into checks and verify that they work
        PrometheusChecks = create_checks_from_iterable(check_query, self.config.metrics)
        return await PrometheusChecks.run(self.config, matching=matching, halt_on=halt_on)
```

Here the `check_query` inner function is going to be used just like earlier
examples that were "checkified" via the `@check` decorator and the
`self.config.metrics` collection is going to be treated like a list of methods
in a `BaseChecks` subclass.

The call to `create_checks_from_iterable` returns a new dynamically created
subclass of `BaseChecks` with `check_` instance methods attached for every item
in the `self.config.metrics` collection.

The `PrometheusChecks` class behaves exactly like a manually coded checks
subclass and can be filtered, etc.

### Checkable Protocol

Protocols are a relatively recent extension to Python that supports *structural
subtyping*. This is basically the idea that a class does not have to explicitly
inherit from another class in order to be considered its subtype. It is an
extension of the concept of duck typing in dynamic languages to the typing
system (sometimes called "Static Duck Typing", see
[https://www.python.org/dev/peps/pep-0544/](PEP 544)).

The `servo.checks.Checkable` protocol defines a single method called `__check__`
that returns a `Check` object. The protocol is used extensively in the internals
but can be used as a public API to provide check implementations for arbitrary
objects.

### Safe and productive by default

The checks subsystem works really hard to make the easy thing delightful and the
wrong impossible. There is extensive enforcement around type hint contracts to
avoid typo bugs. The code is extensively documented and covered with tests.
