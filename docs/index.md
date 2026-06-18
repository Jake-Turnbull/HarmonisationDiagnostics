# DiagnoseHarmonise

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19595960-blue)](https://doi.org/10.5281/zenodo.19595960)

[![PyPI version](https://img.shields.io/pypi/v/DiagnoseHarmonisation.svg)](https://pypi.org/project/DiagnoseHarmonisation/)
[![Python versions](https://img.shields.io/pypi/pyversions/DiagnoseHarmonisation.svg)](https://pypi.org/project/DiagnoseHarmonisation/)
[![License](https://img.shields.io/pypi/l/DiagnoseHarmonisation.svg)](https://pypi.org/project/DiagnoseHarmonisation/)

DiagnoseHarmonise is an **in-development** library for the streamlined application and assessment of harmonisation algorithms at the summary-measure level. It also serves as a centralised location for popular, well-validated harmonisation methods from the literature. The tool is available on [Github](https://github.com/Jake-Turnbull/HarmonisationDiagnostics?tab=readme-ov-file), please consider starring the page to increase the tools visibility.

In an upcoming paper, we plan to demonstrate that systematic evaluation and reporting of different components of batch effects is not only beneficial for choosing an appropriate harmonisation strategy, but essential for evaluating how well harmonisation has worked.

## Support and Contact

If you find any issues or bugs in the code, please raise an issue or contact one of the following:

- **Jake Turnbull**: [jacob.turnbull@ndcn.ox.ac.uk](mailto:jacob.turnbull@ndcn.ox.ac.uk)
- **Gaurav Bhalerao**: [gaurav.bhalerao@ndcn.ox.ac.uk](mailto:gaurav.bhalerao@ndcn.ox.ac.uk)

---

## Overview

This library is intended to support the streamlined analysis and application of harmonisation for MRI data. Consistent reporting of different components of batch differences should be carried out both pre- and post-harmonisation, both to confirm that harmonisation was needed and to verify that it was successful.

While this tool was developed for MRI data, there is no inherent reason it cannot be used in other research scenarios.

The purpose of harmonisation is to remove technical variation driven by differences in data acquisition (e.g. across sites), while preserving meaningful biological signals of interest.

Harmonisation efficacy should therefore be assessed across two broad categories:

1. **Reduction or removal of batch effects**, i.e. unwanted technical differences between datasets.
2. **Preservation of biological signal**, ensuring that meaningful variability is retained.

This library provides a set of functions to assess the severity, nature, and distribution of batch effects across features in multi-batch data. These diagnostics are intended to provide guidance on the most appropriate harmonisation strategy to apply.

Harmonisation is goal-specific, so its integration into experimental design should be carefully considered. Diagnostic reports can serve as a practical method for informing experimental design decisions.

For cross-sectional workflows, the package now supports both a scripted command-line interface and a desktop GUI launcher through `harmdiag`.

For longitudinal workflows, the package now supports a scripted command-line interface through `harmdiag-longitudinal`
