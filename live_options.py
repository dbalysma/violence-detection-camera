"""Command-line options for live_camera.py, kept import-light so they can be unit-tested.

The archived alert log (results/live/live_alerts.csv) is dissertation evidence from the
12-15 Aug 2026 live sessions. A demo recording must never append to it:

- with --record and no --alert-log, alerts go to outputs/demo_alerts.csv (git-ignored);
- with --record, an --alert-log that points at the archived file is refused;
- with no flags at all, the original behaviour is kept and alerts append to the archived path.
"""

import argparse
from pathlib import Path

ARCHIVED_ALERT_LOG = Path("results/live/live_alerts.csv")
DEMO_ALERT_LOG = Path("outputs/demo_alerts.csv")


def build_parser():
    parser = argparse.ArgumentParser(description="Live webcam violence detection with the temporal alert rule")
    parser.add_argument("--no-gemma", action="store_true", help="Skip the Gemma incident report on each alert")
    parser.add_argument(
        "--record", metavar="PATH",
        help="Also save what the window shows (camera image and overlay, no audio) to this .mp4 file. "
             f"Alerts then go to {DEMO_ALERT_LOG} unless --alert-log is given.",
    )
    parser.add_argument(
        "--alert-log", metavar="PATH",
        help=f"CSV file that alerts are appended to. Default: {DEMO_ALERT_LOG} when recording, "
             f"otherwise the original {ARCHIVED_ALERT_LOG}. Recording never writes to {ARCHIVED_ALERT_LOG}.",
    )
    return parser


def is_archived_alert_log(path):
    return Path(path).resolve() == ARCHIVED_ALERT_LOG.resolve()


def resolve_alert_log(alert_log, record):
    """Return the alert-log path to use, or raise ValueError if a recording would touch the archive."""
    if alert_log is None:
        return DEMO_ALERT_LOG if record else ARCHIVED_ALERT_LOG
    path = Path(alert_log)
    if record and is_archived_alert_log(path):
        raise ValueError(
            f"Refusing to append demo alerts to the archived {ARCHIVED_ALERT_LOG}; "
            f"choose another --alert-log, e.g. {DEMO_ALERT_LOG}"
        )
    return path


def parse_live_args(argv=None):
    """Parse live_camera.py's arguments and resolve the alert-log path (relative to the current directory)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.alert_log = resolve_alert_log(args.alert_log, args.record)
    except ValueError as error:
        parser.error(str(error))
    args.record = Path(args.record) if args.record else None
    return args
