import os

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
    async def startup(self) -> None:
        cfg_dict = self.config
        backend_dict = {'url': self.config.optimizer.url,
                        'account': self.config.optimizer.organization,
                        'app_id': self.config.optimizer.name,
                        }
        servo.logger.info(f"Backend config: {backend_dict}")
        servo.logger.info(f"OLAS config: {cfg_dict}")
        backend_dict['auth_token'] = self.config.optimizer.token.get_secret_value()

        # FIXME: Should not be necessary because of servo.connectors.kubernetes
        # There may be a startup dependency race (no long hanging test path atm)
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            kubernetes_asyncio.config.load_incluster_config()
        else:
            await kubernetes_asyncio.config.load_kube_config()

        olas = controller.OLASController(cfg_dict, backend_dict)
        await olas.boot()

        # repeating_task uses a fixed interval yet the interval of update of K8s node metrics may
        # vary slightly thus for the time being the sleep function is done in controller instead
        self.start_repeating_task(
            "OLAS control loop", 0, olas.control_task
        )

    @servo.on_event()
    async def shutdown(self) -> None:
        self.cancel_repeating_tasks()


app = servo.cli.ConnectorCLI(OLASConnector, help="Opsani Learning Autoscaler")


@app.command()
def show_config(
    context: servo.cli.Context,
):
    """
    Display config for OLAS connector
    """
    print(f"Config is: {context.connector.config}")
