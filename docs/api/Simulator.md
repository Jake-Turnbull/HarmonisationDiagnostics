# Simulator

**Simulation utilities used for creating synthetic datasets for diagnostics and testing.**

::: DiagnoseHarmonisation.Simulator
    options:
      filters:
        - "!^_"
      members_order: source

## Examples

Run the top-level simulator script to generate example data used by unit tests and demos:

```bash
python Simulator.py --help
```

The `Simulator` module is also available inside the `DiagnoseHarmonisation` package for programmatic use.
