import asyncio
import os
import sys
import traceback

import kubernetes_asyncio

import servo
from servo.connectors.olas import controller
from servo.connectors.olas.configuration import OLASConfiguration


@servo.metadata(
    description="OLAS Connector for Opsani",
    version="0.9.0",
    homepage="https://github.com/opsani/servox",
    license=servo.License.apache2,
    maturity=servo.Maturity.experimental,
)
class OLASConnector(servo.BaseConnector):
    config: OLASConfiguration

    @servo.on_event()
    def startup(self) -> None:
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)

        cfg_dict = self.config.olas
        backend_dict = {'url': self.config.optimizer.url,
                        'account': self.config.optimizer.organization,
                        'app_id': self.config.optimizer.name,
                        }
        servo.logger.info(f"Backend config: {backend_dict}")
        servo.logger.info(f"OLAS config: {cfg_dict}")
        backend_dict['auth_token'] = self.config.optimizer.token.get_secret_value()

        olas = controller.OLASController(cfg_dict, backend_dict)
        try:
            task = loop.create_task(_olas_task(olas))
            loop.run_until_complete(task)
        except Exception:
            servo.logger.critical(traceback.format_exc())
            task = loop.create_task(olas.client.upload_message(olas.ts, traceback.format_exc()))
            loop.run_until_complete(task)
            sys.exit(1)

    @servo.on_event()
    async def shutdown(self) -> None:
        self.cancel_repeating_tasks()


async def _olas_task(olas):
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        kubernetes_asyncio.config.load_incluster_config()
    else:
        await kubernetes_asyncio.config.load_kube_config()

    await olas.control_task()


app = servo.cli.ConnectorCLI(OLASConnector, help="Opsani Learning Autoscaler")


@app.command()
def show_config(
    context: servo.cli.Context,
):
    """
    Display config for OLAS connector
    """
    print(f"Config is: {context.connector.config}")
