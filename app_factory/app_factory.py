#!/usr/bin/env python
import asyncio
import click
import httpx
import json
import os
import sys
import time
from typing import Callable, List, Optional, Union

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



@click.command()
@click.option('--endpoint', default='api.opsani.com', help='API endpoint to use.') # TODO what is staging url?
@click.option('--account', default='dev.opsani.com', help='Account name to use.')
@click.option('--token', default='./operator/token', help='Location of token to use.', type=click.File())
@click.option('--cluster', default='test', help='Name of cluster_id to use for filtering applications.')
@click.option('--template', default='pn-ocoe', help='Name of template to use when creating new applications.')
@click.option('--target', default=None, help='Desired number of applications; applications will be created and destroyed as needed.', type=click.IntRange(min=0))
@click.option('--interval', default=0, help='Interval on which to add or remove batches of apps.', type=click.IntRange(min=0))
@click.option('--batch', default=1, help='Max number of apps to creat or destroy each interval', type=click.IntRange(min=1))
@click.option('--list', 'list_', is_flag=True)
def sync_app_factory_wrapper(endpoint: str, account: str, token: str, cluster: str, template: str, target: Optional[int], interval: int, batch: int, list_: bool):
    asyncio.run(app_factory(endpoint, account, token, cluster, template, target, interval, batch, list_))

async def app_factory(endpoint: str, account: str, token: str, cluster: str, template: str, target: Optional[int], interval: int, batch: int, list_: bool):
    click.echo("Initializing...")
    async with httpx.AsyncClient(
        headers={
            'Authorization': f'Bearer {token.read()}',
            'Content-Type': 'application/json',
        },
        base_url=f'{endpoint}/accounts/{account}/',
    ) as client:
        # get applications
        click.echo("Getting existing apps...")
        apps: dict = await httpx_req(client.get, f'application-overrides?cluster_id={cluster}&show_all=1', lambda j: j['data'])

        # echo apps if list
        if list_:
            click.echo(apps)

        # nop if no target
        if target is None:
            # click.echo(apps) # should we list by default?
            return
        
        # get latest template id
        click.echo("Getting latest template version...")
        latest_version: str = await httpx_req(client.get, f'templates/{template}', lambda j: j['data']['version'])

        # parse current state, determine updates
        click.echo("Parsing updates...")
        app_keys = sorted(list(apps.keys()))
        cur_num_apps = len(app_keys)
        delta = target - cur_num_apps
        if cur_num_apps > 0:
            click.echo("Validating existing infrastructure...")
            # lowest numbered existing app should provide an accurate representation of the version at the start of the last app-factory run
            low_app_ver = await httpx_req(client.get, f'applications/{app_keys[0]}', lambda j: j['data']['template']['version'])
            latest_create_state = await httpx_req(client.get, f'applications/{app_keys[-1]}', lambda j: j['data']['state'])
            if latest_create_state != 'active':
                click.echo(f"Highest numbered app {app_keys[-1]} in status {latest_create_state}, waiting for 'active' status...")
                await asyncio.gather((lifecycle_status(client, app, 'active') for app in app_keys))

            if low_app_ver != latest_version:
                click.echo(f"App {app_keys[0]} template version {low_app_ver} did not match latest version {latest_version}. Destroying all {cur_num_apps} existing apps...")
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
                    click.echo(f"Hole Detected; Found {len(to_delete)} apps numbered out of sequencing boundaries. Deleting now...")
                    await gather_ops(lambda app: delete_app(client, app), batch, interval, to_delete)

                to_create = list(set(expected_apps) - set(app_keys))
                if to_create:
                    click.echo(f"Hole Detected; Found {len(to_create)} apps missing within sequencing boundaries. Creating now...")
                    await gather_ops(lambda app: create_app(client, app, cluster, template, latest_version), batch, interval, to_create)

        # create to fill the shortage
        if delta > 0:
            to_create = [f'{cluster}{i}' for i in range(cur_num_apps + 1, target + 1)]
            click.echo(f"Creating {delta} apps now...")
            await gather_ops(lambda app: create_app(client, app, cluster, template, latest_version), batch, interval, to_create)

        # cleanup applications in excess of target
        elif delta < 0:
            to_delete: list = [f'{cluster}{i}' for i in range(target + 1, cur_num_apps+1)] # delete apps in excess of target
            click.echo(f"Deleting {abs(delta)} apps now...")
            await gather_ops(lambda app: delete_app(client, app), batch, interval, to_delete)

        # else nop

    sys.exit() # status 0

# if __name__ == '__main__':
#     app_factory()
