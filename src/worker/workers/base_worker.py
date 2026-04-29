from abc import ABC, abstractmethod
from src.shared.core.logger import get_logger

class BaseWorker(ABC):
    """
    Base class for all background workers.
    Provides standard lifecycle hooks: setup -> run -> teardown.
    """
    name: str = "BaseWorker"

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.running = True

    def setup(self):
        """Perform initialization (e.g., connect to DB, queue, etc.)"""
        self.logger.info(f"Setting up worker: {self.name}")

    @abstractmethod
    def run(self):
        """Main execution logic — must be implemented by subclasses"""
        pass

    def teardown(self):
        """Perform cleanup (e.g., close connections)"""
        self.logger.info(f"Tearing down worker: {self.name}")

    def start(self):
        """Entry point to start the worker lifecycle"""
        try:
            self.setup()
            self.run()
        except KeyboardInterrupt:
            self.logger.info(f"Worker {self.name} interrupted by user")
        except Exception as e:
            self.logger.error(f"Worker {self.name} failed with error: {e}", exc_info=True)
            raise
        finally:
            self.teardown()
