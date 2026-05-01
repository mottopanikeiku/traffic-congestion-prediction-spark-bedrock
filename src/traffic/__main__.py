"""Allow `python -m traffic` to invoke the CLI."""
from traffic.cli import main

raise SystemExit(main())
