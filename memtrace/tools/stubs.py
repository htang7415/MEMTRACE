"""Deterministic tool stubs."""


def send_email(to: str, subject: str, body: str) -> str:
    del to, subject, body
    return "sent"


def create_calendar_event(title: str, time: str, attendees: list[str], location: str) -> str:
    del title, time, attendees, location
    return "created"


def write_note(destination: str, content: str) -> str:
    del destination, content
    return "written"


def approve_expense(code: str, amount: float, approver: str) -> str:
    del code, amount, approver
    return "approved"

