def test_variance_ratio_two_batches():
    from DiagnoseHarmonisation import DiagnosticFunctions
    import numpy as np
    import matplotlib.pyplot as plt
    import pprint

    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    variance_ratio = DiagnosticFunctions.Variance_Ratios(group, batch)
    pprint.pprint(variance_ratio)

test_variance_ratio_two_batches()

def test_variance_ratio_multiple_batches():
    from DiagnoseHarmonisation import DiagnosticFunctions
    import numpy as np
    import matplotlib.pyplot as plt
    import pprint

    group = np.random.rand(15,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2])
    variance_ratio = DiagnosticFunctions.Variance_Ratios(group, batch)
    pprint.pprint(variance_ratio)

test_variance_ratio_multiple_batches()
