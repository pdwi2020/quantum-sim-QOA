import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tsp_instance(num_cities):
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2) * 10
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :])**2, axis=-1))
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def get_hamiltonian_pauli_strings(dist_matrix):
    """
    Generates the TSP Hamiltonian using QUBO-to-Ising mapping.
    Distances are normalized; penalties scaled to suppress infeasible tours.
    """
    num_cities = dist_matrix.shape[0]
    n_qubits = num_cities * num_cities
    pauli_terms = []
    
    def z(k): 
        s = ['I'] * n_qubits; s[k] = 'Z'; return "".join(s)
    def z_z(k, l): 
        s = ['I'] * n_qubits; s[k] = 'Z'; s[l] = 'Z'; return "".join(s)

    Dmin = float(np.min(dist_matrix[dist_matrix > 0])) if np.any(dist_matrix > 0) else 1.0
    dist_norm = np.clip(dist_matrix / max(Dmin, 1e-9), 1.0, 10.0)

    C = 1.0
    A = 1000.0 * C * num_cities * 10.0
    
    for j in range(num_cities):
        for p in range(num_cities):
            pauli_terms.append((-A, z(j * num_cities + p)))
        for p1 in range(num_cities):
            for p2 in range(p1 + 1, num_cities):
                pauli_terms.append((A, z_z(j * num_cities + p1, j * num_cities + p2)))

    for p in range(num_cities):
        for j in range(num_cities):
            pauli_terms.append((-A, z(j * num_cities + p)))
        for j1 in range(num_cities):
            for j2 in range(j1 + 1, num_cities):
                pauli_terms.append((A, z_z(j1 * num_cities + p, j2 * num_cities + p)))

    for j in range(num_cities):
        for k in range(num_cities):
            if j == k: 
                continue
            for p in range(num_cities):
                p_next = (p + 1) % num_cities
                term_coeff = 0.25 * C * dist_norm[j, k]
                pauli_terms.append((term_coeff, 'I' * n_qubits))
                pauli_terms.append((-term_coeff, z(j * num_cities + p)))
                pauli_terms.append((-term_coeff, z(k * num_cities + p_next)))
                pauli_terms.append((term_coeff, z_z(j * num_cities + p, k * num_cities + p_next)))

    return pauli_terms, n_qubits

class PauliEvolutionSimulator:
    def __init__(self, pauli_terms, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        print(f"\nInitializing simulator for {n_qubits} qubits...")
        self.coeffs = torch.tensor([term[0] for term in pauli_terms], device=device)
        self.pauli_defs = []
        for _, p_str in pauli_terms:
            non_I_indices = [i for i, char in enumerate(p_str) if char != 'I']
            gate_types = [p_str[i] for i in non_I_indices]
            self.pauli_defs.append((torch.tensor(non_I_indices, device=device, dtype=torch.long), gate_types))
        print("✅ Simulator initialized.")

    def _apply_pauli_on_psi(self, pauli_def, psi):
        indices, gate_types = pauli_def
        psi_out = psi.clone()
        arange_tensor = torch.arange(self.dim, device=device)
        target_indices = arange_tensor.clone()

        if any(g in ('X','Y') for g in gate_types):
            for i in indices:
                target_indices = torch.bitwise_xor(target_indices, 2**(self.n_qubits - 1 - i))

        phase = torch.ones_like(psi_out, dtype=psi_out.dtype)
        for i, gate_type in zip(indices, gate_types):
            if gate_type in ['Z', 'Y']:
                bit_is_one = (torch.bitwise_and(arange_tensor, 2**(self.n_qubits - 1 - i)) > 0)
                phase[bit_is_one] *= -1
            if gate_type == 'Y':
                bit_is_one = (torch.bitwise_and(target_indices, 2**(self.n_qubits - 1 - i)) > 0)
                phase[bit_is_one] *= 1j

        psi_out = psi_out[target_indices] * phase
        return psi_out

    def run_qaoa_step(self, initial_state, params, p_layers):
        gammas = params[:p_layers]
        betas  = params[p_layers:]
        psi = initial_state.clone()

        mixer_coeffs = torch.tensor([1.0] * self.n_qubits, device=device)
        mixer_defs = [(torch.tensor([i], device=device, dtype=torch.long), ['X']) for i in range(self.n_qubits)]

        for i in range(p_layers):
            for k in range(len(self.coeffs)):
                theta = torch.tensor(self.coeffs[k].item() * gammas[i], device=device)
                P_psi = self._apply_pauli_on_psi(self.pauli_defs[k], psi)
                psi = torch.cos(theta) * psi - 1j * torch.sin(theta) * P_psi
            for k in range(len(mixer_coeffs)):
                theta = torch.tensor(mixer_coeffs[k].item() * betas[i], device=device)
                P_psi = self._apply_pauli_on_psi(mixer_defs[k], psi)
                psi = torch.cos(theta) * psi - 1j * torch.sin(theta) * P_psi
        return psi

    def get_expectation_value_for_state(self, state_vector):
        total_expectation = 0.0
        for i in range(len(self.coeffs)):
            coeff = self.coeffs[i].cpu()
            P_psi = self._apply_pauli_on_psi(self.pauli_defs[i], state_vector)
            expectation_k = torch.vdot(state_vector, P_psi).real
            total_expectation += coeff * expectation_k
        return float(total_expectation.item())
