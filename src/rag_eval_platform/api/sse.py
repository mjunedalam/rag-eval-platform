"""Server-sent events: the text/event-stream format used by /query/stream."""


def sse_event(event: str, data: str) -> str:
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"
