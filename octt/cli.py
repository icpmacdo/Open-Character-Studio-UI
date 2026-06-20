"""Command-line entry point.

    octt constitutions            list available personas
    octt show <persona>           print a constitution
    octt models                   list candidate models for the scaling study
"""

from __future__ import annotations

import argparse

from . import constitution, models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="octt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("constitutions", help="list available personas")

    show = sub.add_parser("show", help="print a constitution")
    show.add_argument("persona")

    sub.add_parser("models", help="list candidate scaling-study models")

    args = parser.parse_args(argv)

    if args.command == "constitutions":
        personas = constitution.available()
        print("\n".join(personas) if personas else "(no constitutions yet)")
    elif args.command == "show":
        c = constitution.load(args.persona)
        print(c.text)
    elif args.command == "models":
        for spec in models.CANDIDATES.values():
            marker = "*" if spec.tinker_id in models.SCALING_TRIANGLE else " "
            print(
                f"{marker} {spec.tinker_id:<32} {spec.arch:<5} "
                f"total={spec.total_params_b:>5}B active={spec.active_params_b:>5}B  {spec.note}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
