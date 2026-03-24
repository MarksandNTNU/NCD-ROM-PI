import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from IPython.display import clear_output as clc
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd

mae = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()
mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"
no_format = lambda prob : ("%.2f" % (prob)) + "%"

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def Padding(data, lag):
    '''
    Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
    '''
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out



class BranchTrunkDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for DeepONet or branch/trunk networks.
    
    Each item is ((trunk_input, branch_input), target)
    """
    def __init__(self, trunk_data, branch_data, targets):
        """
        trunk_data: torch.Tensor of shape (N, trunk_features)
        branch_data: torch.Tensor of shape (N, branch_features)
        targets: torch.Tensor of shape (N, output_features)
        """
        assert len(trunk_data) == len(branch_data) == len(targets), \
            "Trunk, branch, and targets must have the same number of samples."
        
        self.X = {0:trunk_data, 1: branch_data}
        self.Y = targets

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        trunk_input = self.X[0][idx]
        branch_input = self.X[1][idx]
        target = self.Y[idx]
        
        return (trunk_input, branch_input), target


def prepare_shred_datasets(snapshots, sensor_positions=None, train_ratio=0.8, lag=30, modes=15, 
                          nsensors=3, device='cpu', verbose=True):
    """
    Prepare datasets specifically for SHRED-ROM training
    
    Parameters:
    -----------
    snapshots : np.ndarray
        Solution snapshots, shape (ntrajectories, nt, nx)
    sensor_positions : list, optional
        Sensor positions, if None uses equally spaced sensors
    train_ratio : float
        Fraction of data for training
    lag : int
        Lag for time series padding
    modes : int
        Number of POD modes
    nsensors : int
        Number of sensors
    device : str
        PyTorch device ('cpu' or 'cuda')
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict : Dictionary containing SHRED datasets and POD information
    """
    
    if verbose:
        print("=" * 50)
        print("PREPARING SHRED-ROM DATASETS")
        print("=" * 50)
    
    # Get dimensions
    ntrajectories, nt, nx = snapshots.shape
    
    if verbose:
        print(f"Dataset dimensions: {ntrajectories} trajectories, {nt} timesteps, {nx} spatial points")
    
    # Train/validation/test split
    np.random.seed(0)
    ntrain = round(train_ratio * ntrajectories)
    
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask != 0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    
    if verbose:
        print(f"Data split: {len(idx_train)} train, {len(idx_valid)} valid, {len(idx_test)} test")
    
    # Prepare sensor positions
    if sensor_positions is None:
        sensor_positions = [int(nx/4), int(nx/2), int(3*nx/4)]
    
    if verbose:
        print(f"Sensor positions: {sensor_positions}")
    
    # Split data
    snapshots_train = snapshots[idx_train].reshape(-1, nx)
    snapshots_valid = snapshots[idx_valid].reshape(-1, nx)
    snapshots_test = snapshots[idx_test].reshape(-1, nx)
    
    # POD decomposition
    if verbose:
        print(f"\nComputing POD with {modes} modes...")
    
    U, S_full, Vt = np.linalg.svd(snapshots_train, full_matrices=False)
    Y = U[:, :modes]
    S = S_full[:modes]
    V = Vt[:modes, :]
    
    # Project snapshots onto POD basis
    snapshots_train_pod = np.dot(snapshots_train, V.T)
    snapshots_valid_pod = np.dot(snapshots_valid, V.T)
    snapshots_test_pod = np.dot(snapshots_test, V.T)
    
    # Compute reconstruction errors
    snapshots_train_recon = np.dot(snapshots_train_pod, V)
    snapshots_valid_recon = np.dot(snapshots_valid_pod, V)
    snapshots_test_recon = np.dot(snapshots_test_pod, V)
    
    train_error = num2p(mre(torch.from_numpy(snapshots_train), torch.from_numpy(snapshots_train_recon)))
    valid_error = num2p(mre(torch.from_numpy(snapshots_valid), torch.from_numpy(snapshots_valid_recon)))
    test_error = num2p(mre(torch.from_numpy(snapshots_test), torch.from_numpy(snapshots_test_recon)))
    
    if verbose:
        print(f"POD reconstruction errors:")
        print(f"  Train: {train_error}")
        print(f"  Valid: {valid_error}")
        print(f"  Test:  {test_error}")
    
    # Scale POD coefficients
    scalerU = MinMaxScaler()
    scalerU = scalerU.fit(snapshots_train_pod)
    U_train_pod = scalerU.transform(snapshots_train_pod).reshape(-1, nt, modes)
    U_valid_pod = scalerU.transform(snapshots_valid_pod).reshape(-1, nt, modes)
    U_test_pod = scalerU.transform(snapshots_test_pod).reshape(-1, nt, modes)
    
    # Prepare sensor measurements
    U_sensor = snapshots[:, :, sensor_positions]
    
    # Convert to PyTorch tensors
    device = torch.device(device)
    
    if verbose:
        print(f"\nPreparing SHRED-ROM datasets...")
    
    train_data_in = Padding(torch.from_numpy(U_sensor[idx_train]), lag).to(device)
    valid_data_in = Padding(torch.from_numpy(U_sensor[idx_valid]), lag).to(device)
    test_data_in = Padding(torch.from_numpy(U_sensor[idx_test]), lag).to(device)
    
    train_data_out = Padding(torch.from_numpy(U_train_pod), 1).squeeze(1).to(device)
    valid_data_out = Padding(torch.from_numpy(U_valid_pod), 1).squeeze(1).to(device)
    test_data_out = Padding(torch.from_numpy(U_test_pod), 1).squeeze(1).to(device)
    
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    
    if verbose:
        print(f"✅ SHRED-ROM dataset preparation complete!")
    
    return {
        'datasets': {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        },
        'pod': {
            'modes': modes,
            'singular_values': S_full,
            'basis_functions': V,
            'scaler': scalerU,
            'reconstruction_errors': {
                'train': train_error,
                'valid': valid_error,
                'test': test_error
            }
        },
        'indices': {
            'train': idx_train,
            'valid': idx_valid,
            'test': idx_test
        },
        'metadata': {
            'ntrajectories': ntrajectories,
            'nt': nt,
            'nx': nx,
            'nsensors': nsensors,
            'sensor_positions': sensor_positions,
            'lag': lag,
            'device': device
        }
    }


def prepare_don_datasets(snapshots, stacked_time, stacked_space, 
                        snapshots_dense=None, stacked_time_dense=None, stacked_space_dense=None,
                        Lx=10*np.pi, Tfinal=20.0, Tfinal_dense=30.0,
                        train_ratio=0.8, lag=30, nsensors=3, sensor_positions=None,
                        device='cpu', verbose=True):
    """
    Prepare datasets specifically for LSTM DeepONet training
    
    Parameters:
    -----------
    snapshots : np.ndarray
        Solution snapshots, shape (ntrajectories, nt, nx)
    stacked_time : np.ndarray
        Time coordinates, shape (ntrajectories, nt, nx)
    stacked_space : np.ndarray
        Space coordinates, shape (ntrajectories, nt, nx)
    snapshots_dense : np.ndarray, optional
        Dense solution snapshots, shape (ntrajectories, nt_dense, nx_dense)
    stacked_time_dense : np.ndarray, optional
        Dense time coordinates, shape (ntrajectories, nt_dense, nx_dense)
    stacked_space_dense : np.ndarray, optional
        Dense space coordinates, shape (ntrajectories, nt_dense, nx_dense)
    Lx : float
        Domain length
    Tfinal : float
        Final time for regular snapshots
    Tfinal_dense : float
        Final time for dense snapshots
    train_ratio : float
        Fraction of data for training
    lag : int
        Lag for time series padding
    nsensors : int
        Number of sensors
    sensor_positions : list, optional
        Sensor positions, if None uses equally spaced sensors
    device : str
        PyTorch device ('cpu' or 'cuda')
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict : Dictionary containing LSTM DeepONet datasets
    """
    
    if verbose:
        print("=" * 50)
        print("PREPARING LSTM DeepONet DATASETS")
        print("=" * 50)
    
    # Get dimensions
    ntrajectories, nt, nx = snapshots.shape
    if snapshots_dense is not None:
        _, nt_dense, nx_dense = snapshots_dense.shape
    else:
        nt_dense, nx_dense = None, None
    
    if verbose:
        print(f"Dataset dimensions:")
        print(f"  Regular: {ntrajectories} trajectories, {nt} timesteps, {nx} spatial points")
        if snapshots_dense is not None:
            print(f"  Dense:   {ntrajectories} trajectories, {nt_dense} timesteps, {nx_dense} spatial points")
    
    # Train/validation/test split
    np.random.seed(0)
    ntrain = round(train_ratio * ntrajectories)
    
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask != 0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    
    if verbose:
        print(f"Data split: {len(idx_train)} train, {len(idx_valid)} valid, {len(idx_test)} test")
    
    # Prepare sensor positions
    if sensor_positions is None:
        sensor_positions = [int(nx/4), int(nx/2), int(3*nx/4)]
    
    if snapshots_dense is not None:
        sensor_positions_dense = [int(nx_dense/4), int(nx_dense/2), int(3*nx_dense/4)]
    else:
        sensor_positions_dense = None
    
    if verbose:
        print(f"Sensor positions: {sensor_positions}")
        if sensor_positions_dense is not None:
            print(f"Dense sensor positions: {sensor_positions_dense}")
    
    # Split data
    snapshots_train = snapshots[idx_train].reshape(-1, nx)
    snapshots_valid = snapshots[idx_valid].reshape(-1, nx)
    snapshots_test = snapshots[idx_test].reshape(-1, nx)
    
    x_train = stacked_space[idx_train]
    x_valid = stacked_space[idx_valid]
    x_test = stacked_space[idx_test]
    
    # Dense data split (if available)
    if snapshots_dense is not None:
        snapshots_train_dense = snapshots_dense[idx_train].reshape(-1, nx_dense)
        snapshots_valid_dense = snapshots_dense[idx_valid].reshape(-1, nx_dense)
        snapshots_test_dense = snapshots_dense[idx_test].reshape(-1, nx_dense)
        
        x_train_dense = stacked_space_dense[idx_train]
        x_valid_dense = stacked_space_dense[idx_valid]
        x_test_dense = stacked_space_dense[idx_test]
    
    # Prepare sensor measurements
    sensor_measurement = snapshots[:, :, sensor_positions]
    
    # Add time as additional sensor input
    m, n, s = sensor_measurement.shape
    sensor_measurement_ext = np.zeros((m, n, s + 1))
    sensor_measurement_ext[:, :, :-1] = sensor_measurement
    sensor_measurement_ext[:, :, -1] = stacked_time[:, :, 0] / Tfinal  # Normalized time
    
    # Dense sensor measurements (if available)
    if snapshots_dense is not None:
        sensor_measurement_dense = snapshots_dense[:, :, sensor_positions_dense]
        m_d, n_d, s_d = sensor_measurement_dense.shape
        sensor_measurement_dense_ext = np.zeros((m_d, n_d, s_d + 1))
        sensor_measurement_dense_ext[:, :, :-1] = sensor_measurement_dense
        sensor_measurement_dense_ext[:, :, -1] = stacked_time_dense[:, :, 0] / Tfinal_dense
    
    # Convert to PyTorch tensors
    device = torch.device(device)
    
    if verbose:
        print(f"\nPreparing LSTM DeepONet datasets...")
    
    # Regular resolution
    train_branch = Padding(torch.from_numpy(sensor_measurement_ext[idx_train]), lag).float().to(device)
    valid_branch = Padding(torch.from_numpy(sensor_measurement_ext[idx_valid]), lag).float().to(device)
    test_branch = Padding(torch.from_numpy(sensor_measurement_ext[idx_test]), lag).float().to(device)
    
    train_trunk = torch.tensor(1/Lx * x_train.reshape(-1, nx, 1), dtype=torch.float32).to(device)
    valid_trunk = torch.tensor(1/Lx * x_valid.reshape(-1, nx, 1), dtype=torch.float32).to(device)
    test_trunk = torch.tensor(1/Lx * x_test.reshape(-1, nx, 1), dtype=torch.float32).to(device)
    
    train_data_out = torch.tensor(snapshots_train.reshape(-1, nx), dtype=torch.float32).view(-1, nx, 1).to(device)
    valid_data_out = torch.tensor(snapshots_valid.reshape(-1, nx), dtype=torch.float32).view(-1, nx, 1).to(device)
    test_data_out = torch.tensor(snapshots_test.reshape(-1, nx), dtype=torch.float32).view(-1, nx, 1).to(device)
    
    train_dataset = BranchTrunkDataset(train_trunk, train_branch, train_data_out)
    valid_dataset = BranchTrunkDataset(valid_trunk, valid_branch, valid_data_out)
    test_dataset = BranchTrunkDataset(test_trunk, test_branch, test_data_out)
    
    # Prepare output dictionary
    result = {
        'datasets': {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        },
        'indices': {
            'train': idx_train,
            'valid': idx_valid,
            'test': idx_test
        },
        'metadata': {
            'ntrajectories': ntrajectories,
            'nt': nt,
            'nx': nx,
            'nsensors': nsensors,
            'sensor_positions': sensor_positions,
            'lag': lag,
            'Lx': Lx,
            'Tfinal': Tfinal,
            'device': device
        }
    }
    
    # Dense datasets (if available)
    if snapshots_dense is not None:
        if verbose:
            print(f"Preparing dense LSTM DeepONet datasets...")
        
        train_branch_dense = Padding(torch.from_numpy(sensor_measurement_dense_ext[idx_train]), lag).float().to(device)
        valid_branch_dense = Padding(torch.from_numpy(sensor_measurement_dense_ext[idx_valid]), lag).float().to(device)
        test_branch_dense = Padding(torch.from_numpy(sensor_measurement_dense_ext[idx_test]), lag).float().to(device)
        
        train_trunk_dense = torch.tensor(1/Lx * x_train_dense.reshape(-1, nx_dense, 1), dtype=torch.float32).to(device)
        valid_trunk_dense = torch.tensor(1/Lx * x_valid_dense.reshape(-1, nx_dense, 1), dtype=torch.float32).to(device)
        test_trunk_dense = torch.tensor(1/Lx * x_test_dense.reshape(-1, nx_dense, 1), dtype=torch.float32).to(device)
        
        train_data_out_dense = torch.tensor(snapshots_train_dense.reshape(-1, nx_dense), dtype=torch.float32).view(-1, nx_dense, 1).to(device)
        valid_data_out_dense = torch.tensor(snapshots_valid_dense.reshape(-1, nx_dense), dtype=torch.float32).view(-1, nx_dense, 1).to(device)
        test_data_out_dense = torch.tensor(snapshots_test_dense.reshape(-1, nx_dense), dtype=torch.float32).view(-1, nx_dense, 1).to(device)
        
        train_dataset_dense = BranchTrunkDataset(train_trunk_dense, train_branch_dense, train_data_out_dense)
        valid_dataset_dense = BranchTrunkDataset(valid_trunk_dense, valid_branch_dense, valid_data_out_dense)
        test_dataset_dense = BranchTrunkDataset(test_trunk_dense, test_branch_dense, test_data_out_dense)
        
        result['datasets_dense'] = {
            'train': train_dataset_dense,
            'valid': valid_dataset_dense,
            'test': test_dataset_dense
        }
        
        result['metadata'].update({
            'nt_dense': nt_dense,
            'nx_dense': nx_dense,
            'sensor_positions_dense': sensor_positions_dense,
            'Tfinal_dense': Tfinal_dense
        })
    
    if verbose:
        print(f"✅ LSTM DeepONet dataset preparation complete!")
        if snapshots_dense is not None:
            print(f"   Includes dense datasets for super-resolution")
    
    return result


# Example usage:
# 
# # For SHRED-ROM only:
# shred_data = prepare_shred_datasets(snapshots, train_ratio=0.8, lag=30, modes=15)
# train_loader = DataLoader(shred_data['datasets']['train'], batch_size=32, shuffle=True)
#
# # For LSTM DeepONet only:
# don_data = prepare_don_datasets(snapshots, stacked_time, stacked_space, 
#                                 snapshots_dense=snapshots_dense,
#                                 stacked_time_dense=stacked_time_dense,
#                                 stacked_space_dense=stacked_space_dense)
# train_loader = DataLoader(don_data['datasets']['train'], batch_size=32, shuffle=True)


    """
    Comprehensive function to prepare datasets for both SHRED-ROM and LSTM DeepONet
    
    Parameters:
    -----------
    snapshots : np.ndarray
        Solution snapshots, shape (ntrajectories, nt, nx)
    stacked_time : np.ndarray
        Time coordinates, shape (ntrajectories, nt, nx)
    stacked_space : np.ndarray
        Space coordinates, shape (ntrajectories, nt, nx)
    snapshots_dense : np.ndarray, optional
        Dense solution snapshots, shape (ntrajectories, nt_dense, nx_dense)
    stacked_time_dense : np.ndarray, optional
        Dense time coordinates, shape (ntrajectories, nt_dense, nx_dense)
    stacked_space_dense : np.ndarray, optional
        Dense space coordinates, shape (ntrajectories, nt_dense, nx_dense)
    Lx : float
        Domain length
    Tfinal : float
        Final time for regular snapshots
    Tfinal_dense : float
        Final time for dense snapshots
    train_ratio : float
        Fraction of data for training
    lag : int
        Lag for time series padding
    modes : int
        Number of POD modes
    nsensors : int
        Number of sensors
    sensor_positions : list, optional
        Sensor positions, if None uses equally spaced sensors
    device : str
        PyTorch device ('cpu' or 'cuda')
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict : Dictionary containing all prepared datasets and metadata
    """
    
    if verbose:
        print("=" * 60)
        print("PREPARING KDV DATASETS FOR SHRED-ROM AND LSTM DeepONet")
        print("=" * 60)
    
    # Get dimensions
    ntrajectories, nt, nx = snapshots.shape
    if snapshots_dense is not None:
        _, nt_dense, nx_dense = snapshots_dense.shape
    else:
        nt_dense, nx_dense = None, None
    
    if verbose:
        print(f"Dataset dimensions:")
        print(f"  Regular: {ntrajectories} trajectories, {nt} timesteps, {nx} spatial points")
        if snapshots_dense is not None:
            print(f"  Dense:   {ntrajectories} trajectories, {nt_dense} timesteps, {nx_dense} spatial points")
    
    # Train/validation/test split
    np.random.seed(0)
    ntrain = round(train_ratio * ntrajectories)
    
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask != 0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    
    if verbose:
        print(f"Data split: {len(idx_train)} train, {len(idx_valid)} valid, {len(idx_test)} test")
    
    # Prepare sensor positions
    if sensor_positions is None:
        sensor_positions = [int(nx/4), int(nx/2), int(3*nx/4)]
    
    if snapshots_dense is not None:
        sensor_positions_dense = [int(nx_dense/4), int(nx_dense/2), int(3*nx_dense/4)]
    
    if verbose:
        print(f"Sensor positions: {sensor_positions}")
        if snapshots_dense is not None:
            print(f"Dense sensor positions: {sensor_positions_dense}")
    
    # Split data
    snapshots_train = snapshots[idx_train].reshape(-1, nx)
    snapshots_valid = snapshots[idx_valid].reshape(-1, nx)
    snapshots_test = snapshots[idx_test].reshape(-1, nx)
    
    t_train = stacked_time[idx_train]
    t_valid = stacked_time[idx_valid]
    t_test = stacked_time[idx_test]
    
    x_train = stacked_space[idx_train]
    x_valid = stacked_space[idx_valid]
    x_test = stacked_space[idx_test]
    
    # Dense data split (if available)
    if snapshots_dense is not None:
        snapshots_train_dense = snapshots_dense[idx_train].reshape(-1, nx_dense)
        snapshots_valid_dense = snapshots_dense[idx_valid].reshape(-1, nx_dense)
        snapshots_test_dense = snapshots_dense[idx_test].reshape(-1, nx_dense)
        
        t_train_dense = stacked_time_dense[idx_train]
        t_valid_dense = stacked_time_dense[idx_valid]
        t_test_dense = stacked_time_dense[idx_test]
        
        x_train_dense = stacked_space_dense[idx_train]
        x_valid_dense = stacked_space_dense[idx_valid]
        x_test_dense = stacked_space_dense[idx_test]
    
    # POD decomposition
    if verbose:
        print(f"\nComputing POD with {modes} modes...")
    
    U, S_full, Vt = np.linalg.svd(snapshots_train, full_matrices=False)
    Y = U[:, :modes]
    S = S_full[:modes]
    V = Vt[:modes, :]
    
    # Project snapshots onto POD basis
    snapshots_train_pod = np.dot(snapshots_train, V.T)
    snapshots_valid_pod = np.dot(snapshots_valid, V.T)
    snapshots_test_pod = np.dot(snapshots_test, V.T)
    
    # Compute reconstruction errors
    snapshots_train_recon = np.dot(snapshots_train_pod, V)
    snapshots_valid_recon = np.dot(snapshots_valid_pod, V)
    snapshots_test_recon = np.dot(snapshots_test_pod, V)
    
    train_error = num2p(mre(torch.from_numpy(snapshots_train), torch.from_numpy(snapshots_train_recon)))
    valid_error = num2p(mre(torch.from_numpy(snapshots_valid), torch.from_numpy(snapshots_valid_recon)))
    test_error = num2p(mre(torch.from_numpy(snapshots_test), torch.from_numpy(snapshots_test_recon)))
    
    if verbose:
        print(f"POD reconstruction errors:")
        print(f"  Train: {train_error}")
        print(f"  Valid: {valid_error}")
        print(f"  Test:  {test_error}")
    
    # Scale POD coefficients
    scalerU = MinMaxScaler()
    scalerU = scalerU.fit(snapshots_train_pod)
    U_train_pod = scalerU.transform(snapshots_train_pod).reshape(-1, nt, modes)
    U_valid_pod = scalerU.transform(snapshots_valid_pod).reshape(-1, nt, modes)
    U_test_pod = scalerU.transform(snapshots_test_pod).reshape(-1, nt, modes)
    
    # Prepare sensor measurements
    sensor_measurement = snapshots[:, :, sensor_positions]
    
    # Add time as additional sensor input
    m, n, s = sensor_measurement.shape
    sensor_measurement_ext = np.zeros((m, n, s + 1))
    sensor_measurement_ext[:, :, :-1] = sensor_measurement
    sensor_measurement_ext[:, :, -1] = stacked_time[:, :, 0] / Tfinal  # Normalized time
    
    # Dense sensor measurements (if available)
    if snapshots_dense is not None:
        sensor_measurement_dense = snapshots_dense[:, :, sensor_positions_dense]
        m_d, n_d, s_d = sensor_measurement_dense.shape
        sensor_measurement_dense_ext = np.zeros((m_d, n_d, s_d + 1))
        sensor_measurement_dense_ext[:, :, :-1] = sensor_measurement_dense
        sensor_measurement_dense_ext[:, :, -1] = stacked_time_dense[:, :, 0] / Tfinal_dense
    
    # Convert to PyTorch tensors
    device = torch.device(device)
    
    # SHRED-ROM datasets
    if verbose:
        print(f"\nPreparing SHRED-ROM datasets...")
    
    U_sensor = snapshots[:, :, sensor_positions]
    
    train_data_in_shred = Padding(torch.from_numpy(U_sensor[idx_train]), lag).to(device)
    valid_data_in_shred = Padding(torch.from_numpy(U_sensor[idx_valid]), lag).to(device)
    test_data_in_shred = Padding(torch.from_numpy(U_sensor[idx_test]), lag).to(device)
    
    train_data_out_shred = Padding(torch.from_numpy(U_train_pod), 1).squeeze(1).to(device)
    valid_data_out_shred = Padding(torch.from_numpy(U_valid_pod), 1).squeeze(1).to(device)
    test_data_out_shred = Padding(torch.from_numpy(U_test_pod), 1).squeeze(1).to(device)
    
    train_dataset_shred = TimeSeriesDataset(train_data_in_shred, train_data_out_shred)
    valid_dataset_shred = TimeSeriesDataset(valid_data_in_shred, valid_data_out_shred)
    test_dataset_shred = TimeSeriesDataset(test_data_in_shred, test_data_out_shred)
    
    # LSTM DeepONet datasets
    if verbose:
        print(f"Preparing LSTM DeepONet datasets...")
    
    # Regular resolution
    train_branch = Padding(torch.from_numpy(sensor_measurement_ext[idx_train]), lag).float().to(device)
    valid_branch = Padding(torch.from_numpy(sensor_measurement_ext[idx_valid]), lag).float().to(device)
    test_branch = Padding(torch.from_numpy(sensor_measurement_ext[idx_test]), lag).float().to(device)
    
    train_trunk = torch.tensor(1/Lx * x_train.reshape(-1, nx, 1), dtype=torch.float32).to(device)
    valid_trunk = torch.tensor(1/Lx * x_valid.reshape(-1, nx, 1), dtype=torch.float32).to(device)
    test_trunk = torch.tensor(1/Lx * x_test.reshape(-1, nx, 1), dtype=torch.float32).to(device)
    
    train_data_out_don = torch.tensor(snapshots_train.reshape(-1, nx), dtype=torch.float32).view(-1, nx, 1).to(device)
    valid_data_out_don = torch.tensor(snapshots_valid.reshape(-1, nx), dtype=torch.float32).view(-1, nx, 1).to(device)
    test_data_out_don = torch.tensor(snapshots_test.reshape(-1, nx), dtype=torch.float32).view(-1, nx, 1).to(device)
    
    train_dataset_don = BranchTrunkDataset(train_trunk, train_branch, train_data_out_don)
    valid_dataset_don = BranchTrunkDataset(valid_trunk, valid_branch, valid_data_out_don)
    test_dataset_don = BranchTrunkDataset(test_trunk, test_branch, test_data_out_don)
    
    # Dense datasets (if available)
    datasets_dense = {}
    if snapshots_dense is not None:
        if verbose:
            print(f"Preparing dense LSTM DeepONet datasets...")
        
        train_branch_dense = Padding(torch.from_numpy(sensor_measurement_dense_ext[idx_train]), lag).float().to(device)
        valid_branch_dense = Padding(torch.from_numpy(sensor_measurement_dense_ext[idx_valid]), lag).float().to(device)
        test_branch_dense = Padding(torch.from_numpy(sensor_measurement_dense_ext[idx_test]), lag).float().to(device)
        
        train_trunk_dense = torch.tensor(1/Lx * x_train_dense.reshape(-1, nx_dense, 1), dtype=torch.float32).to(device)
        valid_trunk_dense = torch.tensor(1/Lx * x_valid_dense.reshape(-1, nx_dense, 1), dtype=torch.float32).to(device)
        test_trunk_dense = torch.tensor(1/Lx * x_test_dense.reshape(-1, nx_dense, 1), dtype=torch.float32).to(device)
        
        train_data_out_dense = torch.tensor(snapshots_train_dense.reshape(-1, nx_dense), dtype=torch.float32).view(-1, nx_dense, 1).to(device)
        valid_data_out_dense = torch.tensor(snapshots_valid_dense.reshape(-1, nx_dense), dtype=torch.float32).view(-1, nx_dense, 1).to(device)
        test_data_out_dense = torch.tensor(snapshots_test_dense.reshape(-1, nx_dense), dtype=torch.float32).view(-1, nx_dense, 1).to(device)
        
        train_dataset_don_dense = BranchTrunkDataset(train_trunk_dense, train_branch_dense, train_data_out_dense)
        valid_dataset_don_dense = BranchTrunkDataset(valid_trunk_dense, valid_branch_dense, valid_data_out_dense)
        test_dataset_don_dense = BranchTrunkDataset(test_trunk_dense, test_branch_dense, test_data_out_dense)
        
        datasets_dense = {
            'train': train_dataset_don_dense,
            'valid': valid_dataset_don_dense,
            'test': test_dataset_don_dense
        }
    
    # Prepare output dictionary
    result = {
        # Data splits
        'indices': {
            'train': idx_train,
            'valid': idx_valid,
            'test': idx_test
        },
        
        # POD decomposition
        'pod': {
            'modes': modes,
            'singular_values': S_full,
            'basis_functions': V,
            'scaler': scalerU,
            'reconstruction_errors': {
                'train': train_error,
                'valid': valid_error,
                'test': test_error
            }
        },
        
        # SHRED-ROM datasets
        'shred': {
            'train': train_dataset_shred,
            'valid': valid_dataset_shred,
            'test': test_dataset_shred
        },
        
        # LSTM DeepONet datasets
        'don': {
            'train': train_dataset_don,
            'valid': valid_dataset_don,
            'test': test_dataset_don
        },
        
        # Dense datasets (if available)
        'don_dense': datasets_dense,
        
        # Metadata
        'metadata': {
            'ntrajectories': ntrajectories,
            'nt': nt,
            'nx': nx,
            'nt_dense': nt_dense,
            'nx_dense': nx_dense,
            'nsensors': nsensors,
            'sensor_positions': sensor_positions,
            'sensor_positions_dense': sensor_positions_dense if snapshots_dense is not None else None,
            'lag': lag,
            'Lx': Lx,
            'Tfinal': Tfinal,
            'Tfinal_dense': Tfinal_dense,
            'device': device
        }
    }
    
    if verbose:
        print(f"\n✅ Dataset preparation complete!")
        print(f"Available datasets: SHRED-ROM, LSTM DeepONet" + (", Dense LSTM DeepONet" if snapshots_dense is not None else ""))
    
    return result


def multiplot(yts, plot, titles = None, fontsize = None, figsize = None, vertical = False, axis = False, save = False, name = "multiplot"):
    """
    Multi plot of different snapshots
    Input: list of snapshots, related plot function, plot options, save option and save path
    """
    
    plt.figure(figsize = figsize)
    for i in range(len(yts)):
        if vertical:
            plt.subplot(len(yts), 1, i+1)
        else:
            plt.subplot(1, len(yts), i+1)
        plot(yts[i])
        plt.title(titles[i], fontsize = fontsize)
        if not axis:
            plt.axis('off')
    
    if save:
    	plt.savefig(name.replace(".png", "") + ".png", transparent = True, bbox_inches='tight')


def trajectory(yt, plot, title = None, fontsize = None, figsize = None, axis = False, save = False, name = 'gif'):
    """
    Trajectory gif
    Input: trajectory with dimension (sequence length, data shape), related plot function for a snapshot, plot options, save option and save path
    """

    arrays = []
        
    for i in range(yt.shape[0]):
        plt.figure(figsize = figsize)
        plot(yt[i])
        plt.title(title, fontsize = fontsize)
        if not axis:
            plt.axis('off')
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
        

def trajectories(yts, plot, titles = None, fontsize = None, figsize = None, vertical = False, axis = False, save = False, name = 'gif'):
    """
    Gif of different trajectories
    Input: list of trajectories with dimensions (sequence length, data shape), plot function for a snapshot, plot options, save option and save path
    """

    arrays = []

    for i in range(yts[0].shape[0]):

        plt.figure(figsize = figsize)
        for j in range(len(yts)):
            if vertical:
                plt.subplot(len(yts), 1, j+1)
            else:
                plt.subplot(1, len(yts), j+1)
            plot(yts[j][i])
            plt.title(titles[j], fontsize = fontsize)
            if not axis:
                plt.axis('off')

        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
