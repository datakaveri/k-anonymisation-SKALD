import logging
import uuid

RUN_ID = str(uuid.uuid4())[:8]

def setup_logging(log_file="log.txt"):
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | run_id=%(run_id)s | %(message)s"
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    root = logging.getLogger("SKALD")
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Inject run_id automatically
    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.run_id = RUN_ID
            return True

    root.addFilter(ContextFilter())
