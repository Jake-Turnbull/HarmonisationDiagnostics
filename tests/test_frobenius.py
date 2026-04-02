# Test Steve assumptions of Frobenius norm calculations

import numpy as np
from numpy.linalg import norm
from scipy.linalg import norm as scipy_norm
from DiagnoseHarmonisation import DiagnosticFunctions  # Assuming the function to test is in mymodule.py
from DiagnoseHarmonisation import PlotDiagnosticResults


matrix = np.array([[1, 2, 3], [4, 5, 6]])
mean_across_columns = np.mean(matrix, axis=0)
print(matrix)
print(mean_across_columns)

def test_frobenius_norm():
    """
    Docstring for test_frobenius_norm
    Making this test as potential issue of the calculation as Cov(X,X) may be close to identity matrix 
    This would make the test wrong

    This script:
        Create dummy data sets, add stuctured noise that differs by batch index
        Project to lower dimension using PCA (20 components)
        Compute covariance matrices for each batch (using only subjects in this batch and the projected data)
        Compute Frobenius norm between covariance matrices of different batches

        Plot covariance matrices as a hatmap for visual inspection (3 batches, 3 heatmaps)

        Verify that in each case Cov(X_all,X_all) is close to identity matrix
        Verify that Cob(X_b,X)b) is not close to identity matrix for each batch (where X_b is subset of X_all with only subjects from batch b)


    """
    data = np.random.rand(1000, 50)  # 100 samples, 50 features
    batch_indices = np.array([0]*330 + [1]*340 + [2]*330)  # 3 batches
    # Add structured noise based on batch
    data[batch_indices == 0] += np.random.normal(0, 0.5, data[batch_indices == 0].shape)
    data[batch_indices == 1] += np.random.normal(1, 0.5, data[batch_indices == 1].shape)
    data[batch_indices == 2] += np.random.normal(2, 0.5, data[batch_indices == 2].shape)
    # Add non batch dependant covariate effects, but have covariates differ by batch (i.e batch 3 older but age effect is linear)
    # Do same for cognitivce score (0-50)
    ages = np.random.randint(20, 70, size=1000)
    cognitive_scores = np.random.randint(0, 50, size=1000)
    data += (ages[:, np.newaxis] * 0.01)  # Age effect
    data += (cognitive_scores[:, np.newaxis] * 0.02)  # Cognitive score effect

    # PCA projection to 20 components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    # Demean the data before PCA to avoid mean differences dominating first PC (i.e don't force PC'S > 1 to be orthogonal to mean)
    # Demean for each column (feature) across all samples
    demeaned_data = data - np.mean(data, axis=0)
    
    data_pca = pca.fit_transform(demeaned_data)
    # Compute covariance matrices for each batch
    cov_matrices = []
    for batch in np.unique(batch_indices):
        batch_data = data_pca[batch_indices == batch]
        cov_matrix = np.cov(batch_data, rowvar=False)
        cov_matrices.append(cov_matrix)
    
    all_data_cov = np.cov(data_pca, rowvar=False)

    # Show the covariance matrices as heatmaps
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, cov_matrix in enumerate(cov_matrices):
        im = axs[i].imshow(cov_matrix, cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Covariance Matrix Batch {i}')
        fig.colorbar(im, ax=axs[i])
    plt.show()

    # Repeat again for the covariance matrix, rescaked by variance of each PC
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, cov_matrix in enumerate(cov_matrices):
        var_PCs = np.var(data_pca[batch_indices == i], axis=0)
        scaled_cov_matrix = cov_matrix.copy()
        for j in range(scaled_cov_matrix.shape[0]):
            scaled_cov_matrix[j, :] /= var_PCs[j]
        im = axs[i].imshow(scaled_cov_matrix, cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Scaled Covariance Matrix Batch {i}')
        fig.colorbar(im, ax=axs[i])
    plt.show()

    # Compute the Frobenius norm between covariance matrices of different batches
    frobenius_norms = []
    for i in range(len(cov_matrices)):
        for j in range(i + 1, len(cov_matrices)):
            frob_norm = norm(cov_matrices[i] - cov_matrices[j], 'fro')
            frobenius_norms.append(frob_norm)
    
    
    # show cov all as heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(all_data_cov, cmap='winter', interpolation='nearest')
    plt.title('Covariance Matrix All Data')
    plt.colorbar()
    # Add the value each covariance (pc1 vs pc2 etc) to the heatmap
    for i in range(all_data_cov.shape[0]):
        for j in range(all_data_cov.shape[1]):
            plt.text(j, i, f"{all_data_cov[i, j]:.2f}", ha='center', va='center', color='w', fontsize=6)
    plt.show()

    # Check that Cov(X_all,X_all) is close to identity matrix for each batch
    identity = np.eye(cov_matrices[0].shape[0])
    assert all_data_cov.shape == identity.shape, "Covariance matrix shape does not match identity matrix shape"
    identity_diff = norm(all_data_cov - identity, 'fro')
    var_PCs = np.var(data_pca, axis=0)
    # Plot variance of each PC
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(var_PCs)), var_PCs)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance')
    plt.title('Variance of Principal Components')
    # add text value of variance above each bar
    for i, v in enumerate(var_PCs):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    plt.show()

    # Test three scenarios: 
    # 1. Fronobenius norm of the covariance of all data
    # 2. Frobenius norm of covariance of all data scaled by variance of each PC
    # 3. Average (mean) variance of each PC
    # Print the three values
    frob_norm_all = norm(all_data_cov, 'fro')

    # Check identity matrix times variance of each PC equals covariance matrix, plot difference
    identity_scaled = identity.copy()
    for i in range(identity.shape[0]):
        identity_scaled[i, :] *= var_PCs[i]
    # Just visualise difference
    diff_matrix = all_data_cov - identity_scaled
    plt.figure(figsize=(6, 5))
    plt.imshow(diff_matrix, cmap='bwr', interpolation='nearest')
    plt.title('Difference Cov(X_all,X_all) - Identity*Var(PCs)')
    plt.colorbar()
    # Add the value each covariance (pc1 vs pc2 etc) to the heatmap
    for i in range(diff_matrix.shape[0]):
        for j in range(diff_matrix.shape[1]):
            plt.text(j, i, f"{diff_matrix[i, j]:.2f}", ha='center', va='center', color='w', fontsize=6)
    plt.show()

    for i in range(identity.shape[0]):
        # Divide each column by the variance of that column of the PCA index to scale the covariance matrix 
        all_data_cov[i, :] /= var_PCs[i]

    frob_norm_scaled = norm(all_data_cov, 'fro')
    print(f"Frobenius norm of scaled Cov(X_all,X_all): {frob_norm_scaled}")
    identity_diff = norm(all_data_cov - identity, 'fro')

        # show cov all as heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(all_data_cov, cmap='winter', interpolation='nearest')
    plt.title('Covariance Matrix All Data')
    plt.colorbar()
    # Add the value each covariance (pc1 vs pc2 etc) to the heatmap
    for i in range(all_data_cov.shape[0]):
        for j in range(all_data_cov.shape[1]):
            plt.text(j, i, f"{all_data_cov[i, j]:.2f}", ha='center', va='center', color='w', fontsize=6)
    plt.show()


  
    plt.ylabel('Value')
    plt.title('Frobenius Norm and Mean Variance Comparison')
    plt.show()

    assert identity_diff < 0.1, f"Cov(X_all,X_all) is not close to identity matrix, difference: {identity_diff}"