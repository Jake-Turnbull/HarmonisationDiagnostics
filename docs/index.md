# DiagnoseHarmonize

DiagnoseHarmonize is an **in-development** library for the streamlined application and assessment of harmonization algorithms at the summary-measure level. It also serves as a centralised location for popular, well-validated harmonization methods from the literature.

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
