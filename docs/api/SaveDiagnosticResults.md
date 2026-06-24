# SaveDiagnosticResults

**Utilities for serialising and saving diagnostic outputs and HTML reports.**

::: DiagnoseHarmonisation.SaveDiagnosticResults
    options:
      filters:
        - "!^_"
      members_order: source

## Usage

Programmatic example:

```python
from DiagnoseHarmonisation.SaveDiagnosticResults import save_report
save_report(report_object, out_path='diagnostic_report.html')
```

This module contains helpers used by the CLI and GUI to persist analysis outputs.
