import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from IPython.display import clear_output as clc
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA

mae = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()
mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()
num2p = lambda prob : ("%.4f" % (100*prob)) + "%"
no_format = lambda prob : ("%.4f" % (prob)) + "%"

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
    
    data_out = np.zeros((data.shape[0] * data.shape[1], lag, data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out


def prepare_shred_datasets(time, snapshots, sensor_positions=None, train_ratio=0.8, lag=50, modes=15, 
                          nsensors=3, device='cpu', verbose=True, random_state=42):
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
        Number of POD modes to retain
    nsensors : int
        Number of sensors
    device : str
        PyTorch device ('cpu' or 'cuda')
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict : Dictionary containing SHRED-ROM datasets, POD  and scaler
    """
    np.random.seed(random_state)
    # Get dimensions
    ntrajectories, nt, nx = snapshots.shape
    # Set up sensor positions
    if sensor_positions is None:
        sensor_positions = [int(nx/4), int(nx/2), int(3*nx/4)]
    
    
    # Reshape data for POD
    snapshots_reshaped = snapshots.reshape(-1, nx)
    
    # Perform POD decomposition
    if verbose:
        print(f"Performing POD decomposition with {modes} modes...")
    
    ntrain = round(0.8 * ntrajectories)

    idx_train = np.random.choice(ntrajectories, size = ntrain, replace = False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask!=0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    snapshots_train = snapshots[idx_train].reshape(-1, nx)
    snapshots_valid = snapshots[idx_valid].reshape(-1, nx)
    snapshots_test = snapshots[idx_test].reshape(-1, nx)

    U, S_full, Vt = randomized_svd(snapshots_train, n_components=modes, random_state=42)
    Y = U[:, :modes]
    S = S_full[:modes]
    V = Vt[:modes, :]

    # Project snapshots onto POD basis
    snapshots_train_pod = np.dot(snapshots_train, V.T)
    snapshots_valid_pod = np.dot(snapshots_valid, V.T)
    snapshots_test_pod = np.dot(snapshots_test, V.T)

    # Reconstruct from POD coefficients
    snapshots_train_recon = np.dot(snapshots_train_pod, V)
    snapshots_valid_recon = np.dot(snapshots_valid_pod, V)
    snapshots_test_recon = np.dot(snapshots_test_pod, V)


    if verbose:
        train_error = num2p(mre(torch.from_numpy(snapshots_train), torch.from_numpy(snapshots_train_recon)))
        valid_error = num2p(mre(torch.from_numpy(snapshots_valid), torch.from_numpy(snapshots_valid_recon)))
        test_error = num2p(mre(torch.from_numpy(snapshots_test), torch.from_numpy(snapshots_test_recon)))

        print(f"Train POD reconstruction error: {train_error} %")
        print(f"Valid POD reconstruction error: {valid_error} %")
        print(f"Test POD reconstruction error: {test_error} %")

    

    scalerU = MinMaxScaler()
    scalerU = scalerU.fit(snapshots_train_pod)
    U_train_pod = scalerU.transform(snapshots_train_pod).reshape(-1, nt, modes)
    U_valid_pod = scalerU.transform(snapshots_valid_pod).reshape(-1, nt, modes)
    U_test_pod = scalerU.transform(snapshots_test_pod).reshape(-1, nt, modes)

    
    U_sensor = snapshots[:, :, sensor_positions]

    train_data_in = Padding(torch.from_numpy(U_sensor[idx_train]), lag).to(device)
    valid_data_in = Padding(torch.from_numpy(U_sensor[idx_valid]), lag).to(device)
    test_data_in = Padding(torch.from_numpy(U_sensor[idx_test]), lag).to(device)


    train_data_out = Padding(torch.from_numpy(U_train_pod), 1).squeeze(1).to(device)    
    valid_data_out = Padding(torch.from_numpy(U_valid_pod), 1).squeeze(1).to(device)
    test_data_out = Padding(torch.from_numpy(U_test_pod), 1).squeeze(1).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    
    return {
        'snapshots':{
            'time': time,
            'train': snapshots_train,
            'valid': snapshots_valid,
            'test': snapshots_test,
        },
        'pod_datasets': {
            'time': time,
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        },
        'POD':{
            'U':U,
            'S_full':S_full, 
            'Vt':Vt
        },
        'scaler':scalerU
    }



def prepare_NCDE_datasets(time, snapshots, sensor_positions=None, train_ratio=0.8, lag=50, modes=15, 
                          nsensors=3, device='cpu', verbose=True, random_state=42):
    """
    Prepare datasets specifically for NCDE training
    
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
        Number of POD modes to retain
    nsensors : int
        Number of sensors
    device : str
        PyTorch device ('cpu' or 'cuda')
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict : Dictionary containing SHRED-ROM datasets, POD  and scaler
    """
    np.random.seed(random_state)
    # Get dimensions
    ntrajectories, nt, nx = snapshots.shape
    # Set up sensor positions
    if sensor_positions is None:
        sensor_positions = [int(nx/4), int(nx/2), int(3*nx/4)]
    
    
    # Reshape data for POD
    snapshots_reshaped = snapshots.reshape(-1, nx)
    
    # Perform POD decomposition
    if verbose:
        print(f"Performing POD decomposition with {modes} modes...")
    
    ntrain = round(0.8 * ntrajectories)

    idx_train = np.random.choice(ntrajectories, size = ntrain, replace = False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask!=0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]
    snapshots_train = snapshots[idx_train].reshape(-1, nx)
    snapshots_valid = snapshots[idx_valid].reshape(-1, nx)
    snapshots_test = snapshots[idx_test].reshape(-1, nx)

    U, S_full, Vt = randomized_svd(snapshots_train, n_components=modes, random_state=42)
    Y = U[:, :modes]
    S = S_full[:modes]
    V = Vt[:modes, :]

    # Project snapshots onto POD basis
    snapshots_train_pod = np.dot(snapshots_train, V.T)
    snapshots_valid_pod = np.dot(snapshots_valid, V.T)
    snapshots_test_pod = np.dot(snapshots_test, V.T)

    # Reconstruct from POD coefficients
    snapshots_train_recon = np.dot(snapshots_train_pod, V)
    snapshots_valid_recon = np.dot(snapshots_valid_pod, V)
    snapshots_test_recon = np.dot(snapshots_test_pod, V)


    if verbose:
        train_error = num2p(mre(torch.from_numpy(snapshots_train), torch.from_numpy(snapshots_train_recon)))
        valid_error = num2p(mre(torch.from_numpy(snapshots_valid), torch.from_numpy(snapshots_valid_recon)))
        test_error = num2p(mre(torch.from_numpy(snapshots_test), torch.from_numpy(snapshots_test_recon)))

        print(f"Train POD reconstruction error: {train_error} %")
        print(f"Valid POD reconstruction error: {valid_error} %")
        print(f"Test POD reconstruction error: {test_error} %")

    

    scalerU = MinMaxScaler()
    scalerU = scalerU.fit(snapshots_train_pod)
    U_train_pod = scalerU.transform(snapshots_train_pod).reshape(-1, nt, modes)
    U_valid_pod = scalerU.transform(snapshots_valid_pod).reshape(-1, nt, modes)
    U_test_pod = scalerU.transform(snapshots_test_pod).reshape(-1, nt, modes)

    
    sensor_measurements = snapshots[:, :, sensor_positions]
    sensor_measurements_ext = np.zeros((sensor_measurements.shape[0], sensor_measurements.shape[1], sensor_measurements.shape[2] + 1))
    sensor_measurements_ext[:, :, 0] = time
    sensor_measurements_ext[:, :, 1:] = sensor_measurements
    train_data_in = Padding(torch.from_numpy(sensor_measurements_ext[idx_train]), lag).to(device)
    valid_data_in = Padding(torch.from_numpy(sensor_measurements_ext[idx_valid]), lag).to(device)
    test_data_in = Padding(torch.from_numpy(sensor_measurements_ext[idx_test]), lag).to(device)


    train_data_out = Padding(torch.from_numpy(U_train_pod), 1).squeeze(1).to(device)    
    valid_data_out = Padding(torch.from_numpy(U_valid_pod), 1).squeeze(1).to(device)
    test_data_out = Padding(torch.from_numpy(U_test_pod), 1).squeeze(1).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    
    return {
        'snapshots':{
            'train': snapshots_train,
            'valid': snapshots_valid,
            'test': snapshots_test,
        },
        'pod_datasets': {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        },
        'POD':{
            'U':U,
            'S_full':S_full, 
            'Vt':Vt
        },
        'scaler':scalerU
    }


def multiplot(yts, plot, titles = None, fontsize = None, figsize = None, vertical = False, axis = False, save = False, name = "multiplot"):
    """
    Multi plot of different snapshots
    Input: list of snapshots, related plot function, plot options, save option and save path
    """
    
    plt.figure(figsize = figsize)
    
    for i, yt in enumerate(yts):
        plt.subplot(1, len(yts), i + 1) if not vertical else plt.subplot(len(yts), 1, i + 1)
        plot(yt, axis = axis)
        if titles != None:
            plt.title(titles[i], fontsize = fontsize)
    plt.tight_layout()
    
    if save:
        plt.savefig(name + ".png", dpi=150, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
        
        
def animate(yts, plot, fps = 5, axis = False, save = False, name = "animation"):
    """
    Create an animated plot of time series data
    """
    
    clc()
    frames = []
    
    for yt in yts:
        fig, ax = plt.subplots()
        plot(yt, axis = axis)
        
        # Convert figure to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    
    if save:
        imageio.mimsave(name + ".gif", frames, fps=fps)
        print(f"Animation saved as {name}.gif")
    
    return frames

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