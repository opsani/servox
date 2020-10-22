from __future__ import annotations

import asyncio
import contextlib
import pathlib
import random
import time
from typing import AsyncGenerator, Callable, List, Optional, Union

import httpx
import pydantic
import typer

import servo
import servo.cli

METRICS = [
    servo.Metric("throughput", servo.Unit.REQUESTS_PER_MINUTE),
    servo.Metric("error_rate", servo.Unit.PERCENTAGE),
]

COMPONENTS = [
    servo.Component(
        "fake-app",
        settings=[
            servo.CPU(
                min=1,
                max=5
            ),
            servo.Memory(
                min=0.25,
                max=8.0,
                step=0.125
            ),
            servo.Replicas(
                min=1,
                max=10
            )
        ]
    )
]

@servo.metadata(
    description="An emulator that pretends take measurements and make adjustments.",
    version="0.5.0",
    homepage="https://github.com/opsani/servox",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.EXPERIMENTAL,
)
class EmulatorConnector(servo.BaseConnector):
    @servo.on_event()
    async def describe(self) -> servo.Description:
        components = await self.components()
        metrics = await self.metrics()

        components_ = await self.components()
        for component_ in components_:
            for setting in component_.settings:
                setting.value = _random_value_for_setting(setting)

        return servo.Description(metrics=metrics, components=components)

    @servo.on_event()
    async def metrics(self) -> List[servo.Metric]:
        return METRICS.copy()

    @servo.on_event()
    async def components(self) -> List[servo.Component]:
        return COMPONENTS.copy()

    @servo.on_event()
    async def measure(
        self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        wait_duration = _random_duration()
        progress = servo.DurationProgress(wait_duration)
        notifier = lambda p: self.logger.info(
            p.annotate(f"sleeping for {wait_duration} to simulate measurement aggregation...", False),
            progress=p.progress,
        )
        await progress.watch(notifier)

        metrics_ = await self.metrics()
        readings = list(map(lambda m: servo.DataPoint(metric=m, value=random.uniform(10, 2000)), metrics_))

        return servo.Measurement(readings=readings)

    @servo.on_event()
    async def adjust(
        self, adjustments: List[servo.Adjustment], control: servo.Control = servo.Control()
    ) -> servo.Description:
        wait_duration = _random_duration()
        progress = servo.DurationProgress(wait_duration)
        notifier = lambda p: self.logger.info(
            p.annotate(f"sleeping for {wait_duration} to simulate adjustment rollout...", False),
            progress=p.progress,
        )
        await progress.watch(notifier)

        components_ = await self.components()
        for component_ in components_:
            for setting in component_.settings:
                setting.value = _random_value_for_setting(setting)

        return servo.Description(components=components_)


def _random_duration() -> servo.Duration:
    seconds = random.randrange(30, 600)
    return servo.Duration(seconds=seconds)

def _random_value_for_setting(setting: servo.Setting) -> Union[str, servo.Numeric]:
    if isinstance(setting, servo.RangeSetting):
        max = int((setting.max - setting.min) / setting.step)
        return random.randint(0, max) * setting.step + setting.min
    elif isinstance(setting, servo.EnumSetting):
        return random.choice(setting.values)
    else:
        raise ValueError(f"unexpected setting: {repr(setting)}")

##
# Optimizer management

class EmulatorContext(pydantic.BaseModel):
    env_domain: str
    org_domain: str
    token: str
    cluster: str
    
    @property
    def base_url(self) -> str:
        return f"https://{self.env_domain}"

cli = servo.cli.ConnectorCLI(EmulatorConnector, help="Emulate servos for testing optimizers")

@cli.callback()
def callback(
    ctx: typer.Context,
    environment: str = typer.Option(
        "api.opsani.com", # api-stage.opsani.com
        "--environment",
        "-e",
        envvar="EMULATOR_ENVIRONMENT",
        show_envvar=True,
        show_default=True,
        metavar="DOMAIN",
        help="Environment to deploy the optimizers into",
    ),
    organization: str = typer.Option(
        "dev.opsani.com",
        "--organization",
        "-o",
        envvar="EMULATOR_ORGANIZATION",
        show_envvar=True,
        show_default=True,
        metavar="DOMAIN",
        help="Domain name of the organization to spawn apps into",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        envvar="EMULATOR_TOKEN",
        show_envvar=True,
        help="API token. Takes precendence over --token-file",
    ),
    token_file: pathlib.Path = typer.Option(
        "./operator/token",
        "--token-file",
        "-f",
        envvar="EMULATOR_TOKEN_FILE",
        show_envvar=True,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        help="File to load the API token from",
    ),
    cluster: str = typer.Option(
        "test",
        envvar="EMULATOR_CLUSTER",
        show_envvar=True,
        show_default=True,
        metavar="CLUSTER",
        help="Name of cluster_id to use for filtering applications.",
    ),
) -> None:
    token_value = token or token_file.read_text().strip()
    emu_context = EmulatorContext(
        env_domain=environment,
        org_domain=organization,
        token=token_value,
        cluster=cluster
    )
    ctx.obj = emu_context

@cli.command(help="List optimizers")
def list_optimizers(ctx: typer.Context) -> None:
    debug(ctx.obj)
    factory = OptimizerFactory(context=ctx.obj)
    optimizers = servo.cli.run_async(factory.list())
    debug(optimizers)

@cli.command(help="Create optimizers")
def create_optimizers(
    ctx: typer.Context,
    name: str = typer.Argument(
        ...,
        help="Name of the optimizer to provision."
    ),
    template: str = typer.Option(
        "live-traffic-opsani-dev",
        "--template",
        "-t",
        show_envvar=True,
        show_default=True,
        metavar="NAME",
        help="Template to use for optimizer configuration.",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        show_envvar=True,
        show_default=True,
        metavar="NAME",
        help="Template to use for optimizer configuration.",
    ),
) -> None:
    factory = OptimizerFactory(context=ctx.obj)
    optimizers = servo.cli.run_async(factory.create(name=name, template=template, version=version))
    debug(optimizers)

@cli.command(help="Delete optimizers")
def delete_optimizers(ctx: typer.Context) -> None:
    factory = OptimizerFactory(context=ctx.obj)
    optimizers = servo.cli.run_async(factory.list())
    debug(optimizers)

@cli.command(help="Scale optimizers to match an objective")
def scale_optimizers(
    ctx: typer.Context,
    template: str = typer.Option(
        "pn-ocoe",
        "--template",
        "-t",
        envvar="EMULATOR_TEMPLATE",
        show_envvar=True,
        show_default=True,
        metavar="NAME",
        help="Name of template to use when creating new applications.",
    ),
    interval_: Optional[str] = typer.Option(
        None,
        "--interval",
        "-i",
        min=0,
        envvar="EMULATOR_INTERVAL",
        show_envvar=True,
        show_default=True,
        help="Interval on which to add or remove batches of apps.",
        metavar="[DURATION]",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        min=1,
        envvar="EMULATOR_BATCH_SIZE",
        show_envvar=True,
        show_default=True,
        help="Max number of apps to create or destroy each interval.",
        metavar="[COUNT]",
    ),
    count: Optional[int] = typer.Argument(
        ...,
        min=0,
        metavar="COUNT",
        help="Desired number of applications; applications will be created and destroyed as needed.",
    )
) -> None:
    servo.types.Duration(interval_) if interval_ else None
    ...

@cli.command(help="Display details about a template")
def show_template(
    ctx: typer.Context,
    name: str = typer.Argument(
        ...,
        metavar="NAME",
        help="Name of template to display.",
    ),
) -> None:
    ...
    factory = OptimizerFactory(context=ctx.obj)
    template = servo.cli.run_async(factory.get_template(name=name))
    debug(template)

class OptimizerFactory(pydantic.BaseModel):
    # TODO: just inline all of these fields...
    context: EmulatorContext
    
    async def list(self) -> dict:        
        async with self._client() as client:
            response = await client.get(f'application-overrides?cluster_id={self.context.cluster}&show_all=1')
            response.raise_for_status()
            return response.json()['data'] # TODO: Load this into a model
    
    # TODO: name and template are ambiguous...
    async def create(self, name: str, *, template: str, version: Optional[str] = None) -> None:
        # TODO: Model and serialize...
        
        if version is None:
            # TODO: model the template
            template_obj = await self.get_template(template)
            version = template_obj["version"]
        
        create_json = {
            "name": name,
            "tags": {
                "cluster_id": self.context.cluster,
            },
            "template": {
                "name": template,
                "version": version,
            },
        }

        async with self._client() as client:
            response = await client.post('applications', json=create_json)
            response.raise_for_status()
            
            await self.wait_for(name, state='active') # TODO: Should be an enum
    
    # TODO: provision several at once?
    async def create_many(self, count: int = 1, batch_size: int = 1, interval: Optional[servo.types.Duration] = None) -> None:
        # TODO: This needs to handle batch + count appropriately.../
        ...
    
    async def wait_for(self, name: str, *, state: str) -> None:
        async with self._client() as client:
            while True: # TODO add timeout, backoff, iterations
                response = await client.get(f'applications/{name}')
                response.raise_for_status()
                
                current_state = response.json()['data']['state']
                if current_state == state:
                    # TODO: Add logging...
                    # print("\n{} is now {}.".format(app, desired_state))
                    break
                # print("Waiting for {} to reach {} state...".format(app, desired_state))
                if state == 'active' and current_state == 'inactive':
                    raise Exception(f'Optimizer {name} became inactive during wait for "active" state')
                await asyncio.sleep(10)
        
    async def delete(self) -> None:
        ...
    
    async def scale(self) -> None:
        ...
    
    async def get_template(self, name: str) -> dict:
        async with self._client() as client:
            response = await client.get(f'templates/{name}')
            response.raise_for_status()
            return response.json()['data'] # TODO: Load this into a model
        latest_version: str = await httpx_req(client.get, f'templates/{template}', lambda j: j['data']['version'])
    
    @contextlib.asynccontextmanager
    async def _client(self) -> AsyncGenerator[httpx.AsyncClient, None, None]:
        async with httpx.AsyncClient(
            headers={
                'Authorization': f'Bearer {self.context.token}',
                'Content-Type': 'application/json',
            },
            base_url=f'{self.context.base_url}/accounts/{self.context.org_domain}/',
        ) as client:
            yield client
        
async def app_factory(endpoint: str, account: str, token: str, cluster: str, template: str, target: Optional[int], interval: int, batch: int, list_: bool):
    typer.echo("Initializing...")
    async with httpx.AsyncClient(
        headers={
            'Authorization': f'Bearer {token.read()}',
            'Content-Type': 'application/json',
        },
        base_url=f'{endpoint}/accounts/{account}/',
    ) as client:
        # get applications
        typer.echo("Getting existing apps...")
        apps: dict = await httpx_req(client.get, f'application-overrides?cluster_id={cluster}&show_all=1', lambda j: j['data'])

        # echo apps if list
        if list_:
            typer.echo(apps)

        # nop if no target
        if target is None:
            # click.echo(apps) # should we list by default?
            return
        
        # get latest template id
        typer.echo("Getting latest template version...")
        latest_version: str = await httpx_req(client.get, f'templates/{template}', lambda j: j['data']['version'])

        # parse current state, determine updates
        typer.echo("Parsing updates...")
        app_keys = sorted(list(apps.keys()))
        cur_num_apps = len(app_keys)
        delta = target - cur_num_apps
        if cur_num_apps > 0:
            typer.echo("Validating existing infrastructure...")
            # lowest numbered existing app should provide an accurate representation of the version at the start of the last app-factory run
            low_app_ver = await httpx_req(client.get, f'applications/{app_keys[0]}', lambda j: j['data']['template']['version'])
            latest_create_state = await httpx_req(client.get, f'applications/{app_keys[-1]}', lambda j: j['data']['state'])
            if latest_create_state != 'active':
                typer.echo(f"Highest numbered app {app_keys[-1]} in status {latest_create_state}, waiting for 'active' status...")
                await asyncio.gather(*(lifecycle_status(client, app, 'active') for app in app_keys))

            if low_app_ver != latest_version:
                typer.echo(f"App {app_keys[0]} template version {low_app_ver} did not match latest version {latest_version}. Destroying all {cur_num_apps} existing apps...")
                # destroy all
                await gather_ops(lambda app: delete_app(client, app), batch, interval, app_keys)

                # create all (not sure if this is too hacky)
                delta = target
                cur_num_apps = 0
            else:
                # check for holes and cleanup
                expected_apps: list = [f'{cluster}{i}' for i in range(1, cur_num_apps+1)]
                to_delete = list(set(app_keys) - set(expected_apps))
                if to_delete:
                    typer.echo(f"Hole Detected; Found {len(to_delete)} apps numbered out of sequencing boundaries. Deleting now...")
                    await gather_ops(lambda app: delete_app(client, app), batch, interval, to_delete)

                to_create = list(set(expected_apps) - set(app_keys))
                if to_create:
                    typer.echo(f"Hole Detected; Found {len(to_create)} apps missing within sequencing boundaries. Creating now...")
                    await gather_ops(lambda app: create_app(client, app, cluster, template, latest_version), batch, interval, to_create)

        # create to fill the shortage
        if delta > 0:
            to_create = [f'{cluster}{i}' for i in range(cur_num_apps + 1, target + 1)]
            typer.echo(f"Creating {delta} apps now...")
            await gather_ops(lambda app: create_app(client, app, cluster, template, latest_version), batch, interval, to_create)

        # cleanup applications in excess of target
        elif delta < 0:
            to_delete: list = [f'{cluster}{i}' for i in range(target + 1, cur_num_apps+1)] # delete apps in excess of target
            typer.echo(f"Deleting {abs(delta)} apps now...")
            await gather_ops(lambda app: delete_app(client, app), batch, interval, to_delete)

####

async def httpx_req(client_method: Callable, url: str, extractor: Callable[[dict], Union[str, dict]], json: Optional[dict] = None) -> Union[str, dict]:
    if json:
        response = await client_method(url=url, json=json)
    else:
        response = await client_method(url=url)

    response.raise_for_status()
    try:
        result = extractor(response.json())
    except KeyError as e:
        raise Exception('Unable to parse result from response JSON: {}'.format(response.json())) from e

    return result

async def lifecycle(client: httpx.AsyncClient, app: str, command: str):
    lifecycle_payload = {"command": command}
    change_state = await httpx_req(client.post, f'applications/{app}/lifecycle', lambda j: j, json=lifecycle_payload)
    return change_state


async def lifecycle_status(client: httpx.AsyncClient, app: str, desired_state: str):
    while True: # TODO timeout?
        app_state = await httpx_req(client.get, f'applications/{app}', lambda j: j['data']['state'])
        if app_state == desired_state:
            # print("\n{} is now {}.".format(app, desired_state))
            break
        # print("Waiting for {} to reach {} state...".format(app, desired_state))
        if desired_state == 'active' and app_state == 'inactive':
            raise Exception(f'App {app} became inactive during wait for "active" status')
        await asyncio.sleep(10)

async def delete_app(client: httpx.AsyncClient, app: str):
    await lifecycle(client, app, 'deactivate')
    await lifecycle_status(client, app, 'inactive')
    
    retries = 3
    captured_error = None
    while retries > 0:
        try:
            await httpx_req(client.delete, f'applications/{app}', lambda j: j)
            return # done if status 200
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503: # 503 - Not all data was removed, operation should be retried.
                captured_error = e
                retries -= 1
                # try again
            else:
                raise
    
    raise Exception(f'Ran out of retries attempting to delete app {app}') from captured_error

# curl -fsS -X POST -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' \
#  -d "{\"name\":\"$1\",\"template\":{\"name\":\"$2\",\"version\":\"${T_VER}\"}}" \
#  https://api.opsani.com/accounts/${ACCOUNT}/applications
async def create_app(client: httpx.AsyncClient, app: str, cluster_id: str, template_name: str, template_version: str):
    create_json = {
        "name": app,
        "tags": {
            "cluster_id": cluster_id,
        },
        "template": {
            "name": template_name,
            "version": template_version,
        },
    }

    await httpx_req(client.post, f'applications', lambda j: j, create_json)
    
    await lifecycle_status(client, app, 'active')

async def gather_ops(op: Callable[[str], None], batch_size: int, batch_interval: int, target_apps: List[str]):
    for i in range(0, len(target_apps), batch_size):
        app_batch = target_apps[i:i+batch_size]
        start_time = time.time()
        await asyncio.gather(*(op(app) for app in app_batch))
        await asyncio.sleep(max(0, batch_interval - (time.time() - start_time))) # TODO should interval waiting only start after creations are completed?
