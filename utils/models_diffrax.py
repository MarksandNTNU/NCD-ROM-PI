import math 
import time 

import diffrax 
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr 
import jax.scipy as jsp

import matplotlib.pyplot as plt
import optax
import jax.nn.initializers as init



def gelu(x):
    """GELU activation function (matches PyTorch's default)."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3)))) 


class Func(eqx.Module):

    mlp: eqx.nn.MLP
    data_size: int 
    hidden_size: int 

    def __init__(self, data_size, hidden_size, width_size, depth, activation, *, key,  **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size = hidden_size, 
            out_size = hidden_size*data_size,
            width_size = width_size, 
            depth = depth, 
            activation = activation,  
            key = key 
        )
    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)

class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear 
    decoder: eqx.nn.Sequential
    use_layer_norm: bool
    
    def __init__(self, data_size, hidden_size, width_size, depth, activation_cde, activation_decoder, output_size, decoder_sizes, *, key, use_layer_norm=False, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, activation = activation_cde, final_activation = activation_cde, key = ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, activation_cde, key = fkey)
        self.linear = eqx.nn.Linear(hidden_size, output_size, key = lkey)
        self.use_layer_norm = use_layer_norm
        
        dkeys = jr.split(key, len(decoder_sizes) + 2)  # +2 for first and last Linear
        layers = []
        
        # First hidden layer
        layers.append(eqx.nn.Linear(hidden_size, decoder_sizes[0], key=dkeys[0]))
        if use_layer_norm:
            layers.append(eqx.nn.LayerNorm(decoder_sizes[0]))
        layers.append(eqx.nn.Lambda(activation_decoder))

        # Middle hidden layers
        for i in range(len(decoder_sizes) - 1):
            layers.append(eqx.nn.Linear(decoder_sizes[i], decoder_sizes[i+1], key=dkeys[i+1]))
            if use_layer_norm:
                layers.append(eqx.nn.LayerNorm(decoder_sizes[i+1]))
            layers.append(eqx.nn.Lambda(activation_decoder))

        # Output layer (no activation, no layer norm)
        layers.append(eqx.nn.Linear(decoder_sizes[-1], output_size, key=dkeys[-1]))
        self.decoder = eqx.nn.Sequential(layers)

    def __call__(self, ts, coeffs):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Heun()
        # Explicit dt0 prevents the PID controller from choosing a microscopic
        # first step when the VF norm is large (main cause of max_steps reached)
        dt0 = (ts[-1] - ts[0]) / 100.0
        y0 = self.initial(control.evaluate(ts[0]))
        saveat = diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            max_steps=int(1e4),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
            saveat=saveat,
            throw=False,  # return best partial solution instead of crashing
        )
        final_hidden = solution.ys[-1] 
        # prediction = self.linear(final_hidden)
        prediction = self.decoder(final_hidden)
        return prediction

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    for array in arrays:
        print(array.shape[0], dataset_size)
    assert all(array.shape[0] == dataset_size for array in arrays)
    indicies = jnp.arange(dataset_size)
    while True: 
        perm = jr.permutation(key, indicies)
        (key,) = jr.split(key, 1)
        start = 0 
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def prepare_data_CDE(X, Y):
    ts = X[:,:, -1]
    ys = X[:,:,:-1]
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    _, _, data_size = ys.shape
    return {'ts': ts, 'Y':Y, 'coeffs':coeffs}, data_size


def loss_mse_CDE(model, ti, y_i, coeff_i):
    preds = jax.vmap(model)(ti, coeff_i)
    return jnp.mean(jnp.sum((preds - y_i)**2, axis = -1))

def loss_rmsre_CDE(model, ti, y_i, coeff_i):
    preds = jax.vmap(model)(ti, coeff_i)
    return jnp.mean(jnp.sqrt(jnp.sum((preds - y_i)**2, axis = -1))/ jnp.sqrt(jnp.sum((y_i**2), axis = -1)))

@eqx.filter_jit
def eval_mse_CDE(model, ti, y_i, coeff_i):
    return loss_mse_CDE(model, ti, y_i, coeff_i)

@eqx.filter_jit 
def make_step_CDE(model, data_i, opt_state, optim):
    ti, y_i, *coeff_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_CDE, has_aux=False)(model, ti, y_i, coeff_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state

def make_warmup_cosine_schedule(lr, warmup_steps, total_steps):
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=warmup_steps,
            ),
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps - warmup_steps,
            ),
        ],
        boundaries=[warmup_steps],
    )


def make_warmup_piecewise_schedule_explicit(
    learning_rates,
    boundaries,
    warmup_steps,
):
    assert len(boundaries) == len(learning_rates), (
        "boundaries must have same length as learning_rates"
    )

    schedules = []
    schedules.append(
        optax.linear_schedule(init_value=0.0, end_value=learning_rates[0], transition_steps=warmup_steps))
    schedules += [optax.constant_schedule(lr) for lr in learning_rates]

    return optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries,
    )



def fit_CDE(model, train_data, valid_data, epochs, batch_size, lr, seed=42, early_stopping=20, verbose=True, grad_clip = 1.0):
    """
    Epoch-based training for Neural CDE model (matches PyTorch behavior).
    
    Each epoch iterates through ALL batches in the training set,
    then evaluates on the FULL validation set once per epoch.
    """
    key = jr.PRNGKey(seed)
    
    # Setup optimizer (no gradient clipping to match PyTorch)
    optim = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    
    # Get dataset info
    n_samples = train_data['ts'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    early_stopping_counter = 0
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        # Shuffle indices at the start of each epoch (like PyTorch DataLoader with shuffle=True)
        key, shuffle_key = jr.split(key)
        indices = jr.permutation(shuffle_key, n_samples)
        
        # Iterate through ALL batches in the dataset
        epoch_train_loss = 0.0
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            ts_batch = train_data['ts'][batch_indices]
            Y_batch = train_data['Y'][batch_indices]
            # Handle coefficients (tuple of arrays)
            coeffs_batch = tuple(c[batch_indices] for c in train_data['coeffs'])
            
            # Gradient step
            train_mse, model, opt_state = make_step_CDE(
                model, (ts_batch, Y_batch) + coeffs_batch, opt_state, optim
            )
            epoch_train_loss += float(train_mse)
        
        # Average training loss over batches
        epoch_train_loss /= n_batches
        
        # Evaluate on FULL validation set (once per epoch, like PyTorch)
        valid_mse = float(eval_mse_CDE(
            model,
            valid_data['ts'],
            valid_data['Y'],
            valid_data['coeffs']
        ))
        
        end = time.time()
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(valid_mse)
        
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_train_loss:.4e} | Valid {valid_mse:.4e} | "
                  f"Time {(end-start):.2f}s | Patience {early_stopping_counter}/{early_stopping}", end='\r')
        
        # Early stopping check
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}, best validation loss: {best_loss:.4e}")
            break
    
    if verbose and early_stopping_counter < early_stopping:
        print(f"\nTraining done: Train loss = {train_losses[-1]:.4e} | Valid loss = {valid_losses[-1]:.4e}")
    
    return best_model, train_losses, valid_losses


class SHRED(eqx.Module):
    """SHRED model with stacked LSTMs and decoder (matches PyTorch implementation)."""
    lstms: tuple
    decoder: eqx.nn.Sequential
    hidden_size: int
    hidden_layers: int
    use_layer_norm: bool

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        decoder_sizes=(350, 400),
        activation=jax.nn.relu,
        *,
        key,
        use_layer_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.use_layer_norm = use_layer_norm
        
        # Split keys for LSTM layers and decoder layers
        num_decoder_layers = len(decoder_sizes) + 1
        keys = jr.split(key, hidden_layers + num_decoder_layers)
        lstm_keys = keys[:hidden_layers]
        dec_keys = keys[hidden_layers:]
        
        # Create stacked LSTM cells (like PyTorch nn.LSTM with num_layers)
        lstms = []
        for i in range(hidden_layers):
            lstm = eqx.nn.LSTMCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                key=lstm_keys[i],
            )
            lstms.append(lstm)
        self.lstms = tuple(lstms)

        # Build decoder: Linear -> LayerNorm -> Activation -> ... -> Linear
        sizes = (hidden_size,) + tuple(decoder_sizes) + (output_size,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=dec_keys[i]))
            # Add LayerNorm and activation to all but the last layer
            if i != len(sizes) - 2:
                if use_layer_norm:
                    layers.append(eqx.nn.LayerNorm(sizes[i + 1]))
                layers.append(eqx.nn.Lambda(activation))

        self.decoder = eqx.nn.Sequential(layers)

    def __call__(self, x):
        """
        x shape: (seq_length, input_size)
        Returns: (output_size,)
        
        Uses jax.lax.scan to efficiently process sequences through stacked LSTMs.
        """
        # Initialize hidden states for all layers: tuple of (h, c) pairs
        init_states = tuple(
            (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
            for _ in range(self.hidden_layers)
        )
        
        def step(states, x_t):
            """
            Process one timestep through all stacked LSTM layers.
            
            Args:
                states: tuple of (h, c) pairs for each layer
                x_t: input at timestep t, shape (input_size,)
            
            Returns:
                new_states: updated (h, c) pairs
                output: output of final LSTM layer
            """
            new_states = []
            layer_input = x_t
            
            # Process through each LSTM layer
            for layer_idx, (lstm, (h, c)) in enumerate(zip(self.lstms, states)):
                h, c = lstm(layer_input, (h, c))
                new_states.append((h, c))
                # Output of this layer becomes input to next layer
                layer_input = h
            
            # Return updated states and the output of the final layer
            return tuple(new_states), layer_input
        
        # Scan over entire sequence
        final_states, outputs = jax.lax.scan(step, init_states, x)
        
        # Get the output from the final timestep of the final LSTM layer
        final_hidden = outputs[-1]
        
        # Pass through decoder
        return self.decoder(final_hidden)

def loss_mse_SHRED(model, S_i, y_i):
    preds = jax.vmap(model)(S_i)
    loss_by_batch = jnp.sum((preds - y_i)**2, axis = -1)
    return jnp.mean(loss_by_batch)

def loss_rmsre_SHRED(model, S_i, y_i):
    preds = jax.vmap(model)(S_i)
    return jnp.mean(jnp.sqrt(jnp.sum((preds - y_i)**2), axis = -1)/ jnp.sqrt(jnp.sum((y_i**2)), axis = -1))

@eqx.filter_jit
def eval_mse_SHRED(model, S_i, y_i):
    return loss_mse_SHRED(model, S_i, y_i)

@eqx.filter_jit
def make_step_SHRED(model, data_i, opt_state, optim):
    S_i, y_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_SHRED, has_aux=False)(model, S_i, y_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state


def fit_SHRED(model, train_data, valid_data, epochs, batch_size, lr, seed=42, early_stopping=20, verbose=True):
    """
    Epoch-based training for SHRED model (matches PyTorch behavior).
    
    Each epoch iterates through ALL batches in the training set, 
    then evaluates on the FULL validation set once per epoch.
    """
    key = jr.PRNGKey(seed)
    
    # Setup optimizer (no gradient clipping to match PyTorch)
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    
    # Get dataset info
    n_samples = train_data['S_i'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    early_stopping_counter = 0
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        # Shuffle indices at the start of each epoch (like PyTorch DataLoader with shuffle=True)
        key, shuffle_key = jr.split(key)
        indices = jr.permutation(shuffle_key, n_samples)
        
        # Iterate through ALL batches in the dataset
        epoch_train_loss = 0.0
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            S_i_batch = train_data['S_i'][batch_indices]
            Y_batch = train_data['Y'][batch_indices]
            
            # Gradient step
            train_mse, model, opt_state = make_step_SHRED(
                model, (S_i_batch, Y_batch), opt_state, optim
            )
            epoch_train_loss += float(train_mse)
        
        # Average training loss over batches
        epoch_train_loss /= n_batches
        
        # Evaluate on FULL validation set (once per epoch, like PyTorch)
        valid_mse = float(eval_mse_SHRED(model, valid_data['S_i'], valid_data['Y']))
        
        end = time.time()
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(valid_mse)
        
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_train_loss:.4e} | Valid {valid_mse:.4e} | "
                  f"Time {(end-start):.2f}s | Patience {early_stopping_counter}/{early_stopping}", end='\r')
        
        # Early stopping check
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}, best validation loss: {best_loss:.4e}")
            break
    
    if verbose and early_stopping_counter < early_stopping:
        print(f"\nTraining done: Train loss = {train_losses[-1]:.4e} | Valid loss = {valid_losses[-1]:.4e}")

    return best_model, train_losses, valid_losses


# ── Weighted losses, step functions and training loops ────────────────────────

def weighted_mse_loss(y_pred, y_true, weights):
    """Weighted MSE: sum over modes weighted by importance, mean over batch."""
    return jnp.mean(jnp.sum((y_pred - y_true) ** 2 * weights[None, :], axis=-1))


def weighted_rmsre_loss(y_pred, y_true, weights):
    """Weighted RMSRE: weighted root-mean-squared relative error."""
    weighted_sq = jnp.sum((y_pred - y_true) ** 2 * weights[None, :], axis=-1)
    norm = jnp.sqrt(jnp.sum(y_true ** 2, axis=-1)) + 1e-8
    return jnp.mean(jnp.sqrt(weighted_sq) / norm)


# NOTE: plain functions (no @eqx.filter_jit) so they can be called inside
# JIT-compiled step functions without creating a double-JIT boundary.
def loss_mse_CDE_weighted(model, ts, coeffs, y, weights):
    return weighted_mse_loss(jax.vmap(model)(ts, coeffs), y, weights)


def loss_mse_SHRED_weighted(model, X, y, weights):
    return weighted_mse_loss(jax.vmap(model)(X), y, weights)


@eqx.filter_jit
def eval_mse_CDE_weighted(model, ts, coeffs, y, weights):
    return loss_mse_CDE_weighted(model, ts, coeffs, y, weights)


@eqx.filter_jit
def eval_mse_SHRED_weighted(model, X, y, weights):
    return loss_mse_SHRED_weighted(model, X, y, weights)


@eqx.filter_jit
def make_step_CDE_weighted(model, ts_batch, coeffs_batch, Y_batch, weights, opt_state, optim):
    """Gradient step for Neural CDE with weighted MSE."""
    mse, grads = eqx.filter_value_and_grad(loss_mse_CDE_weighted)(
        model, ts_batch, coeffs_batch, Y_batch, weights)
    updates, opt_state = optim.update(grads, opt_state)
    return mse, eqx.apply_updates(model, updates), opt_state


@eqx.filter_jit
def make_step_SHRED_weighted(model, S_i_batch, Y_batch, weights, opt_state, optim):
    """Gradient step for SHRED / SHREDAttention with weighted MSE."""
    mse, grads = eqx.filter_value_and_grad(loss_mse_SHRED_weighted)(
        model, S_i_batch, Y_batch, weights)
    updates, opt_state = optim.update(grads, opt_state)
    return mse, eqx.apply_updates(model, updates), opt_state


def fit_CDE_weighted(
    model, train_data, valid_data, weights,
    epochs=100, batch_size=64, lr=1e-2,
    seed=42, early_stopping=20, verbose=True, grad_clip=1.0,
):
    """Epoch-based training for Neural CDE with mode-weighted MSE loss."""
    key = jr.PRNGKey(seed)
    optim = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    train_losses, valid_losses = [], []
    n_samples = train_data['ts'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    patience = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        key, sk = jr.split(key)
        indices = jr.permutation(sk, n_samples)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = indices[b * batch_size: min((b + 1) * batch_size, n_samples)]
            train_mse, model, opt_state = make_step_CDE_weighted(
                model,
                train_data['ts'][idx],
                tuple(c[idx] for c in train_data['coeffs']),
                train_data['Y'][idx],
                weights, opt_state, optim,
            )
            epoch_loss += float(train_mse)
        epoch_loss /= n_batches
        v_loss = float(eval_mse_CDE_weighted(
            model, valid_data['ts'], valid_data['coeffs'], valid_data['Y'], weights))
        train_losses.append(epoch_loss)
        valid_losses.append(v_loss)
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_loss:.4e} | Valid {v_loss:.4e} | "
                  f"Time {time.time()-t0:.2f}s | Patience {patience}/{early_stopping}", end='\r')
        if v_loss < best_loss:
            best_loss, best_model, patience = v_loss, jax.tree_util.tree_map(lambda x: x, model), 0
        else:
            patience += 1
        if patience >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (best valid {best_loss:.4e})")
            break
    else:
        if verbose:
            print(f"\nDone: train {train_losses[-1]:.4e} | valid {valid_losses[-1]:.4e}")
    return best_model, train_losses, valid_losses


def fit_SHRED_weighted(
    model, train_data, valid_data, weights,
    epochs=100, batch_size=64, lr=1e-2,
    seed=42, early_stopping=20, verbose=True, grad_clip=1.0,
):
    """Epoch-based training for SHRED / SHREDAttention with mode-weighted MSE loss."""
    key = jr.PRNGKey(seed)
    optim = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    train_losses, valid_losses = [], []
    n_samples = train_data['S_i'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    patience = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        key, sk = jr.split(key)
        indices = jr.permutation(sk, n_samples)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = indices[b * batch_size: min((b + 1) * batch_size, n_samples)]
            train_mse, model, opt_state = make_step_SHRED_weighted(
                model, train_data['S_i'][idx], train_data['Y'][idx], weights, opt_state, optim)
            epoch_loss += float(train_mse)
        epoch_loss /= n_batches
        v_loss = float(eval_mse_SHRED_weighted(model, valid_data['S_i'], valid_data['Y'], weights))
        train_losses.append(epoch_loss)
        valid_losses.append(v_loss)
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_loss:.4e} | Valid {v_loss:.4e} | "
                  f"Time {time.time()-t0:.2f}s | Patience {patience}/{early_stopping}", end='\r')
        if v_loss < best_loss:
            best_loss, best_model, patience = v_loss, jax.tree_util.tree_map(lambda x: x, model), 0
        else:
            patience += 1
        if patience >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (best valid {best_loss:.4e})")
            break
    else:
        if verbose:
            print(f"\nDone: train {train_losses[-1]:.4e} | valid {valid_losses[-1]:.4e}")
    return best_model, train_losses, valid_losses


class MultiHeadAttention(eqx.Module):
    """Multi-head self-attention layer for processing sequential data."""
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    fc_out: eqx.nn.Linear
    num_heads: int
    embed_dim: int
    head_dim: int
    scale: float
    
    def __init__(self, embed_dim, num_heads, *, key, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        
        q_key, k_key, v_key, out_key = jr.split(key, 4)
        
        self.query = eqx.nn.Linear(embed_dim, embed_dim, key=q_key)
        self.key = eqx.nn.Linear(embed_dim, embed_dim, key=k_key)
        self.value = eqx.nn.Linear(embed_dim, embed_dim, key=v_key)
        self.fc_out = eqx.nn.Linear(embed_dim, embed_dim, key=out_key)
    
    def __call__(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape (seq_length, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor, shape (seq_length, embed_dim)
        """
        seq_length = x.shape[0]
        
        # Linear projections - vmap over sequence dimension
        Q = jax.vmap(self.query)(x)  # (seq_length, embed_dim)
        K = jax.vmap(self.key)(x)    # (seq_length, embed_dim)
        V = jax.vmap(self.value)(x)  # (seq_length, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_length, self.num_heads, self.head_dim)  # (seq_length, num_heads, head_dim)
        K = K.reshape(seq_length, self.num_heads, self.head_dim)
        V = V.reshape(seq_length, self.num_heads, self.head_dim)
        
        # Compute attention scores
        # (seq_length, num_heads, head_dim) @ (head_dim, seq_length) -> (num_heads, seq_length, seq_length)
        scores = jnp.einsum('shd,thd->hst', Q, K) * self.scale
        
        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)  # (num_heads, seq_length, seq_length)
        
        # Apply attention to values
        # (num_heads, seq_length, seq_length) @ (seq_length, num_heads, head_dim) -> (seq_length, num_heads, head_dim)
        attn_output = jnp.einsum('hst,thd->shd', attn_weights, V)  # (seq_length, num_heads, head_dim)
        
        # Reshape back
        attn_output = attn_output.reshape(seq_length, self.embed_dim)  # (seq_length, embed_dim)
        
        # Final linear projection - vmap over sequence dimension
        out = jax.vmap(self.fc_out)(attn_output)
        
        return out


class SHREDAttention(eqx.Module):
    """SHRED model with self-attention layers and FFN (feed-forward network).
    
    Uses multi-head self-attention with feed-forward networks to process sequential data,
    matching the standard Transformer architecture. Each block contains:
    - LayerNorm → MultiHeadAttention → Residual
    - LayerNorm → FFN (MLP) → Residual
    """
    attention_layers: tuple
    fc_layers: tuple
    ffn_layers: tuple
    layer_norms_attn: tuple
    layer_norms_ffn: tuple
    decoder: eqx.nn.Sequential
    hidden_size: int
    num_attention_layers: int
    use_layer_norm: bool
    
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        num_attention_layers=2,
        num_heads=4,
        ffn_width=256,
        decoder_sizes=(350, 400),
        activation=jax.nn.relu,
        *,
        key,
        use_layer_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_layers = num_attention_layers
        self.use_layer_norm = use_layer_norm
        
        # Split keys for all layer types
        # num_attention_layers attention layers
        # num_attention_layers input projection layers
        # num_attention_layers FFN layers
        # 2 * num_attention_layers layer norms (for attention and FFN)
        # num_decoder_layers for decoder
        num_decoder_layers = len(decoder_sizes) + 1
        total_keys_needed = (
            num_attention_layers +  # attention layers
            num_attention_layers +  # input projection layers
            num_attention_layers +  # FFN layers
            (2 * num_attention_layers if use_layer_norm else 0) +  # layer norms
            num_decoder_layers  # decoder layers
        )
        keys = jr.split(key, total_keys_needed)
        
        key_idx = 0
        attn_keys = keys[key_idx:key_idx + num_attention_layers]
        key_idx += num_attention_layers
        
        fc_keys = keys[key_idx:key_idx + num_attention_layers]
        key_idx += num_attention_layers
        
        ffn_keys = keys[key_idx:key_idx + num_attention_layers]
        key_idx += num_attention_layers
        
        if use_layer_norm:
            ln_attn_keys = keys[key_idx:key_idx + num_attention_layers]
            key_idx += num_attention_layers
            ln_ffn_keys = keys[key_idx:key_idx + num_attention_layers]
            key_idx += num_attention_layers
        
        dec_keys = keys[key_idx:key_idx + num_decoder_layers]
        
        # Create attention layers
        attention_layers = []
        for i in range(num_attention_layers):
            attn = MultiHeadAttention(hidden_size, num_heads, key=attn_keys[i])
            attention_layers.append(attn)
        self.attention_layers = tuple(attention_layers)
        
        # Create input projection layers (input to hidden and between layers)
        fc_layers = []
        fc_layers.append(eqx.nn.Linear(input_size, hidden_size, key=fc_keys[0]))
        for i in range(1, num_attention_layers):
            fc_layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=fc_keys[i]))
        self.fc_layers = tuple(fc_layers)
        
        # Create FFN (feed-forward network) layers for each attention block
        # Standard transformer FFN: hidden_size -> ffn_width -> hidden_size
        ffn_layers = []
        for i in range(num_attention_layers):
            ffn = eqx.nn.MLP(
                in_size=hidden_size,
                out_size=hidden_size,
                width_size=ffn_width,
                depth=2,
                activation=activation,
                final_activation=activation,
                key=ffn_keys[i],
            )
            ffn_layers.append(ffn)
        self.ffn_layers = tuple(ffn_layers)
        
        # Create layer norms for transformer-style attention blocks
        if use_layer_norm:
            layer_norms_attn = [eqx.nn.LayerNorm(hidden_size) for _ in range(num_attention_layers)]
            layer_norms_ffn = [eqx.nn.LayerNorm(hidden_size) for _ in range(num_attention_layers)]
            self.layer_norms_attn = tuple(layer_norms_attn)
            self.layer_norms_ffn = tuple(layer_norms_ffn)
        else:
            self.layer_norms_attn = tuple()
            self.layer_norms_ffn = tuple()
        
        # Build decoder: hidden_size -> output_size
        sizes = (hidden_size,) + tuple(decoder_sizes) + (output_size,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=dec_keys[i]))
            # Add LayerNorm and activation to all but the last layer
            if i != len(sizes) - 2:
                if use_layer_norm:
                    layers.append(eqx.nn.LayerNorm(sizes[i + 1]))
                layers.append(eqx.nn.Lambda(activation))
        
        self.decoder = eqx.nn.Sequential(layers)
    
    def __call__(self, x):
        """
        Args:
            x: Input sequence, shape (seq_length, input_size)
            
        Returns:
            Output predictions, shape (output_size,)
        """
        # Project input to hidden dimension
        h = jax.vmap(self.fc_layers[0])(x)  # (seq_length, hidden_size)
        
        # Apply transformer blocks: attention + FFN with residual connections
        for i, attn_layer in enumerate(self.attention_layers):
            # ─── Attention Block ───
            # LayerNorm before attention (pre-norm transformer style)
            if self.use_layer_norm:
                h_norm = jax.vmap(self.layer_norms_attn[i])(h)
            else:
                h_norm = h
            
            # Multi-head attention
            attn_out = attn_layer(h_norm)  # (seq_length, hidden_size)
            
            # Residual connection
            h = h + attn_out  # (seq_length, hidden_size)
            
            # ─── Feed-Forward Network (FFN) Block ───
            # LayerNorm before FFN
            if self.use_layer_norm:
                h_norm = jax.vmap(self.layer_norms_ffn[i])(h)
            else:
                h_norm = h
            
            # FFN: vmap over sequence dimension since it's per-token
            ffn_out = jax.vmap(self.ffn_layers[i])(h_norm)  # (seq_length, hidden_size)
            
            # Residual connection
            h = h + ffn_out  # (seq_length, hidden_size)
            
            # Optional: Apply projection to next layer if available
            if i < len(self.fc_layers) - 1:
                h_proj = jax.vmap(self.fc_layers[i + 1])(h)
                h = h + jax.nn.relu(h_proj)  # Another residual connection
        
        # Pool over sequence: take the last timestep
        # (Alternative pooling strategies: mean, max, learnable pooling)
        final_hidden = h[-1]  # (hidden_size,)
        
        # Pass through decoder
        return self.decoder(final_hidden)


class SimpleMLPModel(eqx.Module):
    """Simple baseline MLP that maps time series features directly to POD modes.
    
    This model takes aggregated/flattened time series input and maps it to POD 
    coefficients. Useful for comparing against Neural CDE approach.
    """
    mlp: eqx.nn.MLP
    
    def __init__(self, input_size, output_size, width_size=256, depth=3, 
                 activation=jax.nn.tanh, *, key, **kwargs):
        """
        Args:
            input_size: Size of aggregated input features (flattened time series data)
            output_size: Size of POD mode coefficients to predict
            width_size: Width of hidden layers
            depth: Number of hidden layers
            activation: Activation function
            key: JAX random key
        """
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,  # Identity for regression output
            key=key
        )
    
    def __call__(self, x):
        """
        Args:
            x: Input features, shape (input_size,) or (batch_size, input_size)
           Returns: POD mode predictions, shape (output_size,) or (batch_size, output_size)
        """
        return self.mlp(x)


def prepare_data_SimpleMLP(X, Y, aggregation='flatten'):
    """Prepare data for simple MLP baseline.
    
    Takes time series data and aggregates it into fixed-size features.
    
    Args:
        X: Input time series, shape (n_samples, n_timesteps, n_features)
        Y: Target POD modes, shape (n_samples, n_modes)
        aggregation: How to aggregate temporal dimension ('last', 'mean', 'flatten')
    
    Returns:
        dict with 'X' and 'Y' keys for training/validation
        input_size: number of features after aggregation
    """
    if aggregation == 'last':
        # Use only the last timestep
        X_agg = X[:, -1, :]  # (n_samples, n_features)
    elif aggregation == 'mean':
        # Average across timesteps
        X_agg = jnp.mean(X[:, :, :-1], axis=1)  # (n_samples, n_features), exclude time column
    elif aggregation == 'flatten':
        # Flatten entire time series (preserves all numerical values)
        n_samples = X.shape[0]
        X_agg = X.reshape(n_samples, -1)  # (n_samples, n_timesteps * n_features)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    input_size = X_agg.shape[-1]
    return {'X': X_agg, 'Y': Y}, input_size


def loss_mse_MLP(model, x_i, y_i):
    """Compute MSE loss for simple MLP."""
    preds = jax.vmap(model)(x_i)
    loss_by_sample = jnp.sum((preds - y_i) ** 2, axis=-1)
    return jnp.mean(loss_by_sample)

def loss_rmsre_MLP(model, x_i, y_i):
    """Compute RMSRE loss for simple MLP."""
    preds = jax.vmap(model)(x_i)
    mse = jnp.sum((preds - y_i) ** 2, axis=-1)
    norm = jnp.sum(y_i ** 2, axis=-1)
    return jnp.mean(jnp.sqrt(mse / (norm + 1e-10)))

@eqx.filter_jit
def eval_mse_MLP(model, x_i, y_i):
    return loss_mse_MLP(model, x_i, y_i)

@eqx.filter_jit
def make_step_MLP(model, data_i, opt_state, optim):
    """Training step for simple MLP."""
    x_i, y_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_MLP, has_aux=False)(model, x_i, y_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state


def loss_mse_SHRED_Attention(model, S_i, y_i):
    """Compute MSE loss for SHREDAttention model."""
    preds = jax.vmap(model)(S_i)
    loss_by_batch = jnp.sum((preds - y_i)**2, axis = -1)
    return jnp.mean(loss_by_batch)

def loss_rmsre_SHRED_Attention(model, S_i, y_i):
    """Compute RMSRE loss for SHREDAttention model."""
    preds = jax.vmap(model)(S_i)
    return jnp.mean(jnp.sqrt(jnp.sum((preds - y_i)**2, axis=-1) / (jnp.sum((y_i**2), axis=-1) + 1e-10)))

@eqx.filter_jit
def eval_mse_SHRED_Attention(model, S_i, y_i):
    return loss_mse_SHRED_Attention(model, S_i, y_i)

@eqx.filter_jit
def make_step_SHRED_Attention(model, data_i, opt_state, optim):
    """Training step for SHREDAttention model."""
    S_i, y_i = data_i
    mse, grads = eqx.filter_value_and_grad(loss_mse_SHRED_Attention, has_aux=False)(model, S_i, y_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return mse, model, opt_state


def fit_SHRED_Attention(model, train_data, valid_data, epochs, batch_size, lr, seed=42, early_stopping=20, verbose=True):
    """
    Epoch-based training for SHREDAttention model.
    
    Each epoch iterates through ALL batches in the training set,
    then evaluates on the FULL validation set once per epoch.
    """
    key = jr.PRNGKey(seed)
    
    # Setup optimizer
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    
    # Get dataset info
    n_samples = train_data['S_i'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    early_stopping_counter = 0
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        # Shuffle indices at the start of each epoch
        key, shuffle_key = jr.split(key)
        indices = jr.permutation(shuffle_key, n_samples)
        
        # Iterate through ALL batches in the dataset
        epoch_train_loss = 0.0
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            S_i_batch = train_data['S_i'][batch_indices]
            Y_batch = train_data['Y'][batch_indices]
            
            # Gradient step
            train_mse, model, opt_state = make_step_SHRED_Attention(
                model, (S_i_batch, Y_batch), opt_state, optim
            )
            epoch_train_loss += float(train_mse)
        
        # Average training loss over batches
        epoch_train_loss /= n_batches
        
        # Evaluate on FULL validation set (once per epoch)
        valid_mse = float(eval_mse_SHRED_Attention(model, valid_data['S_i'], valid_data['Y']))
        
        end = time.time()
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(valid_mse)
        
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_train_loss:.4e} | Valid {valid_mse:.4e} | "
                  f"Time {(end-start):.2f}s | Patience {early_stopping_counter}/{early_stopping}", end='\r')
        
        # Early stopping check
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}, best validation loss: {best_loss:.4e}")
            break
    
    if verbose and early_stopping_counter < early_stopping:
        print(f"\nTraining done: Train loss = {train_losses[-1]:.4e} | Valid loss = {valid_losses[-1]:.4e}")
    
    return best_model, train_losses, valid_losses


def fit_SimpleMLPModel(model, train_data, valid_data, epochs, batch_size, lr, seed=42, early_stopping=20, verbose=True, grad_clip=1.0):

    """Training loop for simple MLP baseline.
    
    Args:
        model: SimpleMLPModel instance
        train_data: dict with 'X' and 'Y' keys
        valid_data: dict with 'X' and 'Y' keys
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        seed: Random seed
        early_stopping: Patience for early stopping
        verbose: Whether to print progress
        grad_clip: Global gradient norm clip
    
    Returns:
        best_model: Model with best validation loss
        train_losses: List of training losses per epoch
        valid_losses: List of validation losses per epoch
    """
    key = jr.PRNGKey(seed)
    
    # Setup optimizer with gradient clipping
    optim = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    train_losses = []
    valid_losses = []
    
    # Get dataset info
    n_samples = train_data['X'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = jax.tree_util.tree_map(lambda x: x, model)
    early_stopping_counter = 0
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Shuffle indices at the start of each epoch
        key, shuffle_key = jr.split(key)
        indices = jr.permutation(shuffle_key, n_samples)
        
        # Iterate through all batches
        epoch_train_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            x_batch = train_data['X'][batch_indices]
            y_batch = train_data['Y'][batch_indices]
            
            # Gradient step
            train_mse, model, opt_state = make_step_MLP(
                model, (x_batch, y_batch), opt_state, optim
            )
            epoch_train_loss += float(train_mse)
        
        # Average training loss over batches
        epoch_train_loss /= n_batches
        
        # Evaluate on full validation set
        valid_mse = float(eval_mse_MLP(model, valid_data['X'], valid_data['Y']))
        
        end_time = time.time()
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(valid_mse)
        
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | Train {epoch_train_loss:.4e} | Valid {valid_mse:.4e} | "
                  f"Time {(end_time-start_time):.2f}s | Patience {early_stopping_counter}/{early_stopping}", 
                  end='\r')
        
        # Early stopping check
        if valid_mse < best_loss:
            best_loss = valid_mse
            best_model = jax.tree_util.tree_map(lambda x: x, model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}, best validation loss: {best_loss:.4e}")
            break
    
    if verbose and early_stopping_counter < early_stopping:
        print(f"\nTraining done: Train loss = {train_losses[-1]:.4e} | Valid loss = {valid_losses[-1]:.4e}")
    
    return best_model, train_losses, valid_losses
