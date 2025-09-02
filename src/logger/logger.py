from abc import abstractmethod

class AbstractLogger:
    @abstractmethod
    def log_info(self, msg: str) -> None:
        pass

    @abstractmethod
    def log_warning(self, msg: str) -> None:
        pass
    
    @abstractmethod
    def log_error(self, msg: str) -> None:
        pass
    
    def log(self, msg: str, level: str = "info") -> None:
        pass

class Logger(AbstractLogger):
    def log_info(self, msg: str) -> None:
        """Log an informational message."""
        print(f"[INFO] {msg}")
    
    def log_warning(self, msg: str) -> None:
        """Log a warning message."""
        print(f"[WARNING] {msg}")
    
    def log_error(self, msg: str) -> None:
        """Log an error message."""
        print(f"[ERROR] {msg}")

    def _log(self, msg: str, level: str = "info") -> None:
        """Log a message at the specified level, default is 'info'."""
        method_name = f"log_{level}"

        if not hasattr(self, method_name):
            raise ValueError(f"Invalid log level: {level}")
        
        getattr(self, method_name)(msg)

    def log(self, msg: str, level: str = "info") -> None:
        """Log a message at the specified level."""
        self._log(msg, level)