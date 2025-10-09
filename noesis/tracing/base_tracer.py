from abc import ABC, abstractmethod

class Tracer(ABC):
    @abstractmethod
    def register(self):
        """Attach hooks/callbacks."""
        ...

    @abstractmethod
    def unregister(self):
        """Remove hooks/callbacks."""
        ...

    @abstractmethod
    def summary(self) -> dict:
        """Return a structured dict with trace metadata and records."""
        ...
