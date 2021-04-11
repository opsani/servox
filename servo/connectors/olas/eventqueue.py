import asyncio


class EventQueue:
    def __init__(self):
        self.queues = {}
        # queues is a map from event type to a queue list.

    def put_event(self, event_type, event):
        for queue in self.queues.get(event_type, []):
            queue.put_nowait((event_type, event))

    def subscribe_event(self, event_type):
        queue = asyncio.Queue()
        self.queues.setdefault(event_type, []).append(queue)
        return queue


eventQueue = EventQueue()
