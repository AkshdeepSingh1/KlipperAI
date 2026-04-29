import time
from src.worker.workers.base_worker import BaseWorker

class P2TextToContentWorker(BaseWorker):
    """
    Scaffold for the P2 Text-to-Content worker.
    Prints hello world as a proof of concept for the new scalable structure.
    """
    name: str = "P2TextToContentWorker"

    def run(self):
        self.logger.info("Hello World! P2TextToContentWorker is up and running.")
        self.logger.info("This is a web-job style worker scaffold.")
        
        # Example of a periodic task loop
        count = 0
        while self.running and count < 3:
            self.logger.info(f"Processing batch {count}...")
            time.sleep(1)
            count += 1
            
        self.logger.info("Work cycle completed.")
