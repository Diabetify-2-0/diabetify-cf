from __future__ import annotations

import logging
import signal
import sys

from diabetify_cf.config import Settings
from diabetify_cf.engine import build_counterfactual_engine
from diabetify_cf.messaging.rabbitmq_service import RabbitMQCFService


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level.upper())
    logger = logging.getLogger("diabetify_cf")
    engine = build_counterfactual_engine(settings)
    service = RabbitMQCFService(settings=settings, engine=engine)

    def _handle_shutdown(signum: int, _frame: object) -> None:
        logger.info("Received signal=%s. Initiating shutdown.", signum)
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    logger.info("Starting diabetify-cf service in env=%s", settings.app_env)
    service.start()


if __name__ == "__main__":
    main()
