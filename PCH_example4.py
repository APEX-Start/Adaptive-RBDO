# Because we can not get the true constraint functions, we just use the CFD simulation data to estimate the constraint functions.
# this is a interactive function to get the simulation data from the user, but the algorithm is the same as the one in the paper

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyDOE2 import lhs
import time
from scipy.stats import norm
import matplotlib as mpl
import platform
import numdifftools as nd
import GPy # Import GPy

# Global variable to store interactively obtained simulation data
simulation_database = {}

def adaptive_gradient(x, model, rel_tol=1e-8, abs_tol=1e-8, max_iter=15):
    """
    Adaptive gradient calculation using numdifftools
    """
    def scalar_func(x_vec):
        mean, _ = model.predict(x_vec.reshape(1, -1))
        return mean[0, 0]

    gradient_calculator = nd.Gradient(
        scalar_func,
        method='central',
        order=4,
        step=abs_tol,
        num_extrap=max_iter,
        step_ratio=1.5,
        step_nom=np.maximum(np.abs(x), 1e-6)
    )
    return np.atleast_1d(gradient_calculator(x))

def find_mpp_improved(g_model, d_mean_vars, sigma_vars, beta_t, method='hybrid', max_tries=5):
    """
    Search for Most Probable Point (MPP) using AMV method - fixed COBYLA equality constraint issue
    """
    n_dv = len(d_mean_vars)
    u_k = np.zeros(n_dv)
    max_iter_mpp = 10

    for k in range(max_iter_mpp):
        # x_k_eval represents a realization in X-space: X = mu_X + Sigma * U
        x_k_eval = d_mean_vars + u_k * sigma_vars
        
        g_value_mean, _ = g_model.predict(x_k_eval.reshape(1, -1))
        g_value = g_value_mean[0, 0]
        
        # Gradient of g with respect to X, evaluated at x_k_eval
        grad_g_x = adaptive_gradient(x_k_eval, g_model) 
        # Gradient of g with respect to U: grad_g_u = grad_g_x * dX/dU = grad_g_x * sigma_vars
        grad_g_u = grad_g_x * sigma_vars
        
        if np.linalg.norm(grad_g_u) < 1e-10:
            print(f"MPP search for model {g_model.name if hasattr(g_model, 'name') else 'unknown'} converged early due to small gradient norm.")
            break
            
        alpha_mpp = -grad_g_u / np.linalg.norm(grad_g_u)
        beta_est = (g_value - np.dot(grad_g_u, u_k) + np.linalg.norm(grad_g_u) * np.linalg.norm(u_k) * np.dot(alpha_mpp, u_k/ (np.linalg.norm(u_k) + 1e-9) ) ) / np.linalg.norm(grad_g_u)
        beta_hlrf = (np.dot(grad_g_u, u_k) - g_value) / np.linalg.norm(grad_g_u)

        u_next = beta_hlrf * alpha_mpp
        
        if np.linalg.norm(u_next - u_k) < 1e-4:
            u_k = u_next
            break
        # Damping factor for stability
        u_k = 0.5 * u_k + 0.5 * u_next
    
    beta = np.linalg.norm(u_k)
    mpp = d_mean_vars + u_k * sigma_vars
    
    g_value_mean_final, _ = g_model.predict(mpp.reshape(1, -1))
    g_value_final = g_value_mean_final[0,0]
    
    # If g(MPP) is not close to 0, try refinement
    if abs(g_value_final) > 0.1:
        try:
            # Method 1: Use SLSQP (supports equality constraints)
            def distance_func(u_opt):
                return np.linalg.norm(u_opt)
            
            def g_constraint_for_opt(u_opt):
                x_opt = d_mean_vars + u_opt * sigma_vars
                mean_constr, _ = g_model.predict(x_opt.reshape(1, -1))
                return mean_constr[0,0]
            
            # Try using SLSQP
            try:
                result = minimize(distance_func, u_k, method='SLSQP', 
                                 constraints={'type': 'eq', 'fun': g_constraint_for_opt},
                                 options={'maxiter': 100, 'ftol': 1e-6, 'disp': False})
                
                if result.success:
                    u_star = result.x
                    g_final_check = g_constraint_for_opt(u_star)
                    
                    if abs(g_final_check) < abs(g_value_final) and abs(g_final_check) < 0.1:
                        beta = np.linalg.norm(u_star)
                        mpp = d_mean_vars + u_star * sigma_vars
                        g_value_final = g_final_check
                        print(f"MPP refined by SLSQP: New beta = {beta:.4f}, g(MPP) = {g_value_final:.6f}")
                
            except Exception as e:
                # If SLSQP fails, try COBYLA with inequality constraints
                print(f"SLSQP refinement failed: {e}. Attempting with inequality constraints method...")
                
                # Method 2: Convert equality constraint to two inequality constraints |g(x)| <= epsilon
                epsilon = 1e-4
                constraints = [
                    {'type': 'ineq', 'fun': lambda u: epsilon - g_constraint_for_opt(u)},
                    {'type': 'ineq', 'fun': lambda u: g_constraint_for_opt(u) + epsilon}
                ]
                
                result = minimize(distance_func, u_k, method='COBYLA',
                                 constraints=constraints,
                                 options={'maxiter': 100, 'rhobeg': 0.1, 'disp': False})
                
                if result.success:
                    u_star = result.x
                    g_final_check = g_constraint_for_opt(u_star)
                    
                    if abs(g_final_check) < abs(g_value_final):
                        beta = np.linalg.norm(u_star)
                        mpp = d_mean_vars + u_star * sigma_vars
                        g_value_final = g_final_check
                        print(f"MPP refined by COBYLA (inequality constraints): New beta = {beta:.4f}, g(MPP) = {g_value_final:.6f}")
                        
        except Exception as e:
            print(f"MPP refinement completely failed: {e}")
            pass
    
    print(f"AMV method for model {g_model.name if hasattr(g_model, 'name') else 'unknown'}: β = {beta:.4f}, g(MPP) = {g_value_final:.6f}")
    
    return mpp, beta

def initial_samples(bounds, num_samples):
    dim = bounds.shape[0]
    samples = lhs(dim, samples=num_samples, criterion='centermaximin')
    return bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])

def objective_function(d):
    """
    Calculates the weight objective function
    d = [d1, d2, L]
    """
    d1, d2, L = d[0], d[1], d[2]
    rho1, rho2 = 7930, 8000  # Density kg/m³
    
    # Convert to meters
    d1_m = d1 * 1e-3
    d2_m = d2 * 1e-3
    L_m = L * 1e-3
    
    # Calculate weight (kg)
    weight = rho1 * np.pi * ((30e-3)**2 - d2_m**2) * (L_m/4) + rho2 * np.pi * d1_m**2 * (L_m/4)
    return weight

def get_simulation_result_interactive(x_point):
    """
    Interactively obtains simulation results
    """
    global simulation_database
    
    # Convert point to tuple for dictionary key
    point_key = tuple(np.round(x_point, 6))
    
    # Check if data for this point already exists
    if point_key in simulation_database:
        return simulation_database[point_key]
    
    print(f"\n=== New simulation result needed ===")
    print(f"Design point: [d1={x_point[0]:.4f}, d2={x_point[1]:.4f}, L={x_point[2]:.2f}]")
    print("Please perform simulation and enter the results:")
    
    try:
        p_outlet = float(input("Enter P_outlet (Pa): "))
        m_outlet = float(input("Enter M_outlet (kg/s): "))
        
        result = np.array([p_outlet, m_outlet])
        simulation_database[point_key] = result
        
        print(f"Simulation results recorded: P_outlet = {p_outlet:.6f} Pa, M_outlet = {m_outlet:.6f} kg/s")
        return result
        
    except ValueError:
        print("Invalid input format, please re-enter numerical values")
        return get_simulation_result_interactive(x_point)
    except KeyboardInterrupt:
        print("\nUser interruption")
        return np.array([0.0, 0.0])

def constraint_functions_interactive(x):
    """
    Calculates constraint function values based on interactive simulation results
    """
    # Get simulation results
    sim_result = get_simulation_result_interactive(x)
    p_outlet, m_outlet = sim_result[0], sim_result[1]
    
    # Calculate constraint functions
    g1 = p_outlet - 3000000 # g1 = P_outlet - 3000000
    g2 = 0.015 + m_outlet  # g2 = 0.015 + M_outlet
    
    return np.array([g1, g2])

def initialize_with_existing_data():
    """
    Initializes the simulation database with predefined experimental data
    """
    global simulation_database
    
    # Predefined 30 sets of experimental data
    X_samples = np.array([
        [4.7167, 6.245, 251.67],
        [5.0833, 5.675, 261.67],
        [5.0167, 6.485, 248.33],
        [5.1167, 5.735, 205.00],
        [5.0500, 5.975, 235.00],
        [4.7833, 6.365, 201.67],
        [4.9167, 5.705, 245.00],
        [4.6167, 6.185, 268.33],
        [4.6833, 5.645, 298.33],
        [4.9833, 5.825, 291.67],
        [5.4500, 6.005, 231.67],
        [5.3500, 5.615, 255.00],
        [5.1833, 6.305, 285.00],
        [5.3167, 6.035, 265.00],
        [4.6500, 5.795, 278.33],
        [5.1500, 6.155, 211.67],
        [4.9500, 5.855, 215.00],
        [5.4833, 6.065, 288.33],
        [4.8833, 5.915, 238.33],
        [4.5833, 6.335, 241.67],
        [4.8167, 5.945, 228.33],
        [5.2167, 6.095, 275.00],
        [5.2833, 6.125, 281.67],
        [4.5167, 6.455, 295.00],
        [5.3833, 6.215, 258.33],
        [4.7500, 5.885, 221.67],
        [4.5500, 5.765, 218.33],
        [4.8500, 6.395, 271.67],
        [5.4167, 6.425, 225.00],
        [5.2500, 6.275, 208.33]
    ])
    
    # Corresponding simulation results [P_outlet (Pa), M_outlet (kg s^-1)]
    simulation_results = np.array([
        [20333.0, -0.057371],
        [28580.0, -0.0018043],
        [32789.0, -0.055301],
        [36600.0, -0.0034666],
        [33282.0, -0.013703],
        [0.0, -0.072722],
        [31426.0, -0.0068829],
        [0.0, -0.057580],
        [-80367.0, -0.010570],
        [32322.0, -0.0065172],
        [33392.0, -0.0018969],
        [16849.0, -0.00012161],
        [-8.2693E+05, -0.020271],
        [30052.0, -0.0046638],
        [-1.9834E+06, -0.022047],
        [36329.0, -0.021117],
        [36466.0, -0.014301],
        [27678.0, -0.0015233],
        [32996.0, -0.018575],
        [0.0, -0.077907],
        [34291.0, -0.024421],
        [30169.0, -0.0088134],
        [36647.0, -0.0074179],
        [-9.8311E+05, -0.112780],
        [32146.0, -0.008325],
        [35122.0, -0.025269],
        [35911.0, -0.028642],
        [30520.0, -0.056201],
        [36751.0, -0.020554],
        [37516.0, -0.022895]
    ])
    
    # Add data to the simulation database
    for i, x_point in enumerate(X_samples):
        point_key = tuple(np.round(x_point, 6))
        simulation_database[point_key] = simulation_results[i]
    
    print(f"Initialized {len(X_samples)} pre-defined simulation data points to database")
    return X_samples, simulation_results

def batch_constraint_evaluation(X_batch):
    """
    Evaluates constraint functions in batch (for initialization and known data)
    """
    results = []
    for x_point in X_batch:
        # Get simulation results (from database if exists, else interactive input)
        sim_result = get_simulation_result_interactive(x_point)
        p_outlet, m_outlet = sim_result[0], sim_result[1]
        
        # Calculate constraint functions
        g1 = p_outlet  # g1 = P_outlet
        g2 = 0.02 + m_outlet  # g2 = 0.02 + M_outlet
        
        results.append([g1, g2])
    
    return np.array(results)

def check_feasibility(beta_values, beta_target, kriging_models, design_point, mpp_points, e1=None, e2=None):
    """
    Checks the feasibility status of constraints
    """
    feasibility_status = []
    for i, beta in enumerate(beta_values):
        if e1 is None and i < len(mpp_points) and mpp_points[i] is not None:
            _, var_mpp = kriging_models[i].predict(mpp_points[i].reshape(1, -1))
            std_mpp = np.sqrt(var_mpp)
            e1_value = std_mpp[0, 0]
        else:
            e1_value = 0.05
        
        if e2 is None:
            _, var_design = kriging_models[i].predict(design_point.reshape(1, -1))
            std_design = np.sqrt(var_design)
            e2_value = std_design[0, 0]
        else:
            e2_value = e2
        
        threshold = (1 + e1_value * e2_value)  * beta_target
        
        if beta > 1.2 * threshold:
            feasibility_status.append(0)
        elif 1 <= beta / threshold <=  1.2:
            feasibility_status.append(1)
        else:
            feasibility_status.append(2)
    return feasibility_status

def improved_ef_function(x, kriging_model, z_bar=0, epsilon=0.01):
    """
    Calculates the improved EF function value
    """
    X_samples = kriging_model.X 
    if X_samples.shape[0] == 0:
        d_min = 1.0
    else:
        distances = np.sqrt(np.sum((X_samples - x.reshape(1, -1))**2, axis=1))
        d_min = np.min(distances) if distances.size > 0 else 1.0

    u_G_mean, sigma_G_var = kriging_model.predict(x.reshape(1, -1))
    u_G = u_G_mean[0, 0]
    sigma_G = np.sqrt(sigma_G_var[0, 0])
    sigma_G = np.maximum(sigma_G, 1e-6)

    z_plus = z_bar + epsilon
    z_minus = z_bar - epsilon

    Z = np.clip((z_bar - u_G) / sigma_G, -8, 8) 
    Z_plus = np.clip((z_plus - u_G) / sigma_G, -8, 8)
    Z_minus = np.clip((z_minus - u_G) / sigma_G, -8, 8)

    phi_term1 = norm.cdf(Z)
    phi_term2 = norm.cdf(Z_minus)
    phi_term3 = norm.cdf(Z_plus)
    pdf_term1 = norm.pdf(Z)
    pdf_term2 = norm.pdf(Z_minus)
    pdf_term3 = norm.pdf(Z_plus)

    term1 = d_min * (u_G - z_bar) * (2 * phi_term1 - phi_term2 - phi_term3)
    term2 = -d_min * sigma_G * (2 * pdf_term1 - pdf_term2 - pdf_term3)
    term3 = d_min * epsilon * (phi_term3 - phi_term2)
    ef_value = term1 + term2 + term3
    return ef_value

def calculate_sampling_radius(beta_target, kriging_model, design_point, sigma_val, num_test_points=100):
    """
    Calculates the sampling region radius
    """
    dim = len(design_point)
    
    test_points = []
    for _ in range(num_test_points):
        u_sphere = np.random.randn(dim)
        u_sphere = u_sphere / np.linalg.norm(u_sphere) * beta_target
        x_test = design_point + sigma_val * u_sphere
        test_points.append(x_test)
    test_points = np.array(test_points)
    
    y_pred_mean, y_var = kriging_model.predict(test_points)
    y_std = np.sqrt(y_var)
    
    uncertainty_ratio = np.mean(y_std / (np.abs(y_pred_mean) + 1e-10 * np.max(np.abs(y_pred_mean)) + 1e-10 ))
    
    base_radius_x_space = beta_target * sigma_val
    uncertainty_factor = 1.5 + uncertainty_ratio
    
    max_allowable_radius = 0.2 * np.mean(np.abs(design_point))
    radius = min(base_radius_x_space * uncertainty_factor, max_allowable_radius)
    radius = max(radius, 0.01 * sigma_val)

    return radius

def calculate_space_filling_score(point, existing_points, all_samples):
    if all_samples.shape[0] == 0:
        min_dist_all = 1.0
    else:
        distances_to_all = np.linalg.norm(all_samples - point.reshape(1, -1), axis=1)
        min_dist_all = np.min(distances_to_all) if distances_to_all.size > 0 else 1.0
    
    if not existing_points:
        min_dist_current = 1.0
    else:
        existing_points_arr = np.array(existing_points)
        distances_to_current = np.linalg.norm(existing_points_arr - point.reshape(1, -1), axis=1)
        min_dist_current = np.min(distances_to_current) if distances_to_current.size > 0 else 1.0
        
    return 0.7 * min_dist_all + 0.3 * min_dist_current

def check_min_distance(point, existing_points, min_distance, all_samples):
    all_distances = []
    if all_samples.shape[0] > 0:
        all_distances.extend(list(np.linalg.norm(all_samples - point.reshape(1, -1), axis=1)))
    if existing_points:
        existing_points_arr = np.array(existing_points)
        all_distances.extend(list(np.linalg.norm(existing_points_arr - point.reshape(1, -1), axis=1)))
    
    if not all_distances: return True
    return all(d >= min_distance - 1e-9 for d in all_distances if d > 1e-9)

def select_new_samples(feasibility_status, kriging_models, bounds_sel, design_point, 
                       beta_target, sigma_val_sel, num_samples_per_iter=3, max_total_samples=60):
    """
    Selects new sample points
    """
    dim_sel = len(design_point)
    current_total = len(kriging_models[0].X)
    if current_total >= max_total_samples:
        print("Maximum number of samples reached.")
        return np.array([])
    
    w1, w2 = 0.2, 0.8
    design_space_ranges = bounds_sel[:, 1] - bounds_sel[:, 0]
    min_distance = 0.01 * np.mean(design_space_ranges)

    selected_points = []
    all_samples_sel = kriging_models[0].X.copy() if kriging_models[0].X.shape[0] > 0 else np.empty((0, dim_sel))
    
    active_constraints_indices = [i for i, status in enumerate(feasibility_status) if status in [1, 2]]
    
    num_candidates_base = 50 * dim_sel
    
    candidates = []
    if not active_constraints_indices:
        print("Generating candidate points near MPP of all constraints")
        for i, model in enumerate(kriging_models):
            mpp, _ = find_mpp_improved(model, design_point, np.full(dim_sel, sigma_val_sel), beta_target)
            if mpp is None: continue
            for _ in range(max(10, num_candidates_base // len(kriging_models) // 2)): 
                r_mpp = np.random.uniform(0.1, 0.5) * beta_target * sigma_val_sel
                direction_mpp = np.random.randn(dim_sel)
                if np.linalg.norm(direction_mpp) < 1e-9: continue
                direction_mpp /= np.linalg.norm(direction_mpp)
                x_cand = np.clip(mpp + r_mpp * direction_mpp, bounds_sel[:, 0], bounds_sel[:, 1])
                candidates.append(x_cand)
    else:
        print(f"Generating candidate points in the region of active/violated constraints {active_constraints_indices}")
        for i in active_constraints_indices:
            model = kriging_models[i]
            radius = calculate_sampling_radius(beta_target, model, design_point, sigma_val_sel, num_test_points=50)
            mpp, _ = find_mpp_improved(model, design_point, np.full(dim_sel, sigma_val_sel), beta_target)
            if mpp is None: continue

            for _ in range(max(20, num_candidates_base // len(active_constraints_indices))): 
                alpha_interp = np.random.uniform(0.2, 1.5)
                interpolated_point = design_point + alpha_interp * (mpp - design_point)
                
                r_active = radius * np.random.uniform(0.2, 0.8)
                direction_active = np.random.randn(dim_sel)
                if np.linalg.norm(direction_active) < 1e-9: continue
                direction_active /= np.linalg.norm(direction_active)
                perturbation = r_active * direction_active
                x_cand = np.clip(interpolated_point + perturbation, bounds_sel[:, 0], bounds_sel[:, 1])
                candidates.append(x_cand)
    
    if not candidates:
        print("Failed to generate candidate points.")
        return np.array([])
    
    candidates = np.array(list(set(map(tuple, candidates))))
    if candidates.shape[0] == 0:
        print("Candidate list is empty (after deduplication).")
        return np.array([])

    for s_idx in range(num_samples_per_iter):
        if candidates.shape[0] == 0: 
            print(f"Candidates exhausted, {len(selected_points)} points selected.")
            break 
        
        ef_values = []
        for point in candidates:
            total_ef = 0
            target_model_indices = active_constraints_indices if active_constraints_indices else range(len(kriging_models))
            for model_idx in target_model_indices:
                ef = improved_ef_function(point, kriging_models[model_idx])
                total_ef += ef
            ef_values.append(total_ef / max(1, len(target_model_indices)))

        ef_min, ef_max = (np.min(ef_values), np.max(ef_values)) if ef_values else (0,0)
        ef_range = ef_max - ef_min + 1e-10
        
        scores = []
        for i, point in enumerate(candidates):
            current_batch_plus_archive = np.vstack((all_samples_sel, np.array(selected_points))) if selected_points else all_samples_sel
            
            space_score = calculate_space_filling_score(point, selected_points, current_batch_plus_archive)
            
            if check_min_distance(point, selected_points, min_distance, current_batch_plus_archive):
                ef_normalized = (ef_values[i] - ef_min) / ef_range
                space_normalized = space_score / (min_distance + 1e-9) 
                total_score = w1 * ef_normalized + w2 * space_normalized
            else:
                total_score = -np.inf
            scores.append(total_score)
        
        if not scores or np.all(np.isneginf(scores)):
            print("All candidate points fail minimum distance check or have no valid score.")
            break 
            
        best_idx = np.argmax(scores)
        best_point = candidates[best_idx]
        selected_points.append(best_point)
        
        distances_to_best = np.linalg.norm(candidates - best_point.reshape(1, -1), axis=1)
        candidates = candidates[distances_to_best > (min_distance / 2.0)]
        
        print(f"Selecting new sample point {s_idx+1}/{num_samples_per_iter}: {np.round(best_point,5)}, Score: {scores[best_idx]:.4f}")
    
    return np.array(selected_points)

def mcs_reliability_analysis(x_mcs, kriging_models, sigma_val_mcs, num_samples=100000, beta_target=2.0):
    """
    MCS Analysis
    """
    # For a 3D problem, sigma_val_mcs should be an array with 3 elements
    if np.isscalar(sigma_val_mcs):
        # Set standard deviations based on problem definition
        sigma_array = np.array([0.025, 0.025, 0.5])  # [d1_std, d2_std, L_std]
    else:
        sigma_array = sigma_val_mcs
    
    dim = len(x_mcs)
    num_samples = int(num_samples)
    u_samples = np.random.randn(num_samples, dim)
    failure_counts = np.zeros(len(kriging_models))
    
    for i, model in enumerate(kriging_models):
        x_samples_eval = x_mcs.reshape(1, -1) + u_samples * sigma_array
        g_values_mean, _ = model.predict(x_samples_eval)
        failure_counts[i] = np.sum(g_values_mean < 0)
    
    pf_values = failure_counts / num_samples
    pf_values = np.clip(pf_values, 1e-12, 1.0 - 1e-12) 
    beta_values = -norm.ppf(pf_values)
    is_feasible = np.all(beta_values >= beta_target)
    
    return beta_values, pf_values, is_feasible

def create_mcs_constraint_functions(models, beta_t_mcs, sigma_val_c, num_samples=10000):
    """
    Creates MCS-based constraint functions
    """
    constraints = []
    for i, model in enumerate(models):
        def constraint_func(x_constr, model_idx=i):
            beta_values, _, _ = mcs_reliability_analysis(
                x_constr, [models[model_idx]], sigma_val_c, 
                num_samples=num_samples, beta_target=beta_t_mcs)
            return beta_values[0] - (beta_t_mcs + 1e-6)
        constraints.append({'type': 'ineq', 'fun': constraint_func})
    return constraints

def enforce_bounds(x, bounds):
    """
    Enforces boundary constraints
    """
    x_corrected = np.copy(x)
    violations = []
    
    for i in range(len(x)):
        original_val = x[i]
        x_corrected[i] = np.clip(x[i], bounds[i,0], bounds[i,1])
        
        if x_corrected[i] != original_val:
            violations.append(f"x{i+1}: {original_val:.6f} -> {x_corrected[i]:.6f} "
                            f"[{bounds[i,0]}, {bounds[i,1]}]")
    
    if violations:
        print(f"Boundary violation correction: {'; '.join(violations)}")
    
    return x_corrected

def two_stage_optimization(design_point, kriging_models, bounds_opt, 
                           beta_target_opt, sigma_array_opt, sigma_val_opt, use_mcs=True):
    """
    Two-stage optimization
    """
    dim_opt = len(design_point)

    def constraint_violation(x_viol):
        violations = []
        for model in kriging_models:
            pred_mean, pred_var = model.predict(x_viol.reshape(1, -1))
            pred_std = np.sqrt(pred_var)
            conservative_pred = pred_mean[0,0] - 0.5 * pred_std[0,0]
            violations.append(-conservative_pred)
        return np.maximum(0, violations)
    
    def objective_with_penalty(x_pen):
        obj = objective_function(x_pen)
        violations = constraint_violation(x_pen)
        penalty_factor = 10.0 * max(1.0, abs(obj)) / (np.sum(violations**2) + 1e-6) if violations.any() else 10.0
        penalty_factor = min(penalty_factor, 1e6)
        penalty = penalty_factor * np.sum(violations**2)
        return obj + penalty
    
    print("Optimization Stage 1: Finding a point close to feasible region using penalty function")
    result1 = minimize(objective_with_penalty, design_point, 
                       bounds=[(bounds_opt[j, 0], bounds_opt[j, 1]) for j in range(dim_opt)],
                       method='COBYLA',
                       options={'maxiter': 100 * dim_opt, 'rhobeg': 0.05, 'disp': False, 'tol': 1e-3})
    
    intermediate_point = result1.x
    violations_interm = constraint_violation(intermediate_point)
    is_feasible_interm = np.all(violations_interm <= 1e-2)
    
    if is_feasible_interm:
        print(f"Intermediate point constraint violation: {violations_interm}, Objective value: {objective_function(intermediate_point):.4f}")
        starting_point = intermediate_point
    else:
        print(f"Warning: Stage 1 intermediate point does not satisfy constraints, violation: {violations_interm}. Attempting Stage 2 from original design point.")
        starting_point = 0.5 * design_point + 0.5 * intermediate_point
    
    print(f"Optimization Stage 2: Starting explicit constrained optimization from point {np.round(starting_point,5)}")
    
    optimizer_method = 'SLSQP'
    optimizer_options = {'maxiter': 150 * dim_opt, 'ftol': 1e-5, 'disp': False}

    if use_mcs:
        print("Using Monte Carlo simulation for reliability assessment in constraints")
        num_samples_check = int(max(10000, 50 / (norm.cdf(-beta_target_opt)+1e-9)))
        beta_values_start, _, is_starting_feasible = mcs_reliability_analysis(
            starting_point, kriging_models, sigma_val_opt, 
            num_samples=num_samples_check,
            beta_target=beta_target_opt)
        print(f"MCS reliability index at Stage 2 starting point: {np.round(beta_values_start,5)}, Is feasible: {is_starting_feasible}")
        
        mcs_samples_for_constr = int(max(5000, 50 / (norm.cdf(-beta_target_opt)+1e-9)))
        mcs_samples_for_constr = min(mcs_samples_for_constr, 30000)
        print(f"Building constraints with {mcs_samples_for_constr} MCS samples")

        constraints_opt = create_mcs_constraint_functions(kriging_models, beta_target_opt, sigma_val_opt, 
                                                          num_samples=mcs_samples_for_constr)
        optimizer_method = 'COBYLA'
        optimizer_options = {'maxiter': 100 * dim_opt, 'rhobeg': 0.02, 'disp': False, 'catol': 0.005}
        print(f"Optimizer for Stage 2 switched to COBYLA due to MCS constraints, catol={optimizer_options['catol']}")
    else:
        def create_amv_constraint_functions(models_amv, beta_t_amv, sigma_arr_amv):
            constraints_list = []
            for i, model_amv in enumerate(models_amv):
                def constraint_func_amv(x_amv, model_idx=i):
                    try:
                        _, beta_amv = find_mpp_improved(models_amv[model_idx], x_amv, sigma_arr_amv, beta_t_amv)
                        return beta_amv - (beta_t_amv + 1e-6)
                    except Exception as e_amv:
                        print(f"MPP calculation failed in AMV constraint for model {model_idx}: {e_amv}. Using proxy value.")
                        pred_mean_amv, pred_var_amv = models_amv[model_idx].predict(x_amv.reshape(1, -1))
                        if pred_mean_amv[0,0] < 0: return -10.0
                        return pred_mean_amv[0,0] / (np.sqrt(pred_var_amv[0,0]) + 1e-6) - 1.0
                constraints_list.append({'type': 'ineq', 'fun': constraint_func_amv})
            return constraints_list
        constraints_opt = create_amv_constraint_functions(kriging_models, beta_target_opt, sigma_array_opt)
    
    for i_model_sc, _ in enumerate(kriging_models):
        def simple_backup_constraint(x_sc, model_idx_local=i_model_sc):
            pred_mean_sc, pred_var_sc = kriging_models[model_idx_local].predict(x_sc.reshape(1, -1))
            pred_std_sc = np.sqrt(pred_var_sc[0,0])
            return pred_mean_sc[0,0] - 0.5 * pred_std_sc
        constraints_opt.append({'type': 'ineq', 'fun': simple_backup_constraint})
    
    result2 = minimize(objective_function, starting_point, 
                      method=optimizer_method,
                      bounds=[(bounds_opt[j, 0], bounds_opt[j, 1]) for j in range(dim_opt)],
                      constraints=constraints_opt,
                      options=optimizer_options)
    final_point = enforce_bounds(result2.x, bounds_opt)
    is_feasible_final = False 
    
    print("Verifying final point of Stage 2 with high-sample MCS...")
    final_beta_values, final_pf_values, final_is_feasible_mcs = mcs_reliability_analysis(
        final_point, kriging_models, sigma_val_opt, 
        num_samples=100000,
        beta_target=beta_target_opt)
    
    print(f"High-fidelity MCS reliability index for final point ({np.round(final_point,5)}): {np.round(final_beta_values,5)}")
    print(f"  Objective function value: {objective_function(final_point):.4f}, Satisfies MCS requirement: {final_is_feasible_mcs}")
    
    new_design_point = final_point
    is_feasible_final = final_is_feasible_mcs

    if not final_is_feasible_mcs:
        print(f"Warning: Stage 2 optimization result does not meet high-fidelity MCS reliability requirement. Attempting to revert from {optimizer_method} result.")
        print(f"Proceeding with {optimizer_method} result ({np.round(new_design_point,5)}) for further processing.")

    return new_design_point, is_feasible_final

def check_robust_convergence(design_point, previous_design, objective_current, objective_previous, 
                           beta_current, beta_previous, beta_target_conv, mcs_beta=None,
                           design_tol=1e-3, obj_tol=1e-3, beta_tol=0.05):
    """
    Robust convergence criterion
    """
    design_change_abs = np.linalg.norm(design_point - previous_design)
    design_change_rel = design_change_abs / (np.linalg.norm(previous_design) + 1e-9)
    design_converged = design_change_rel < design_tol

    obj_change_abs = abs(objective_current - objective_previous)
    obj_change_rel = obj_change_abs / (abs(objective_previous) + 1e-9)
    obj_converged = obj_change_rel < obj_tol
    
    beta_rel_changes = []
    beta_converged_iter = []
    beta_satisfied_iter = []
    
    is_initial_iteration = beta_previous is None or (isinstance(beta_previous, np.ndarray) and beta_previous.size == 0)
    
    for i in range(len(beta_current)):
        if is_initial_iteration:
            b_prev = beta_target_conv
        else:
            b_prev = beta_previous[i]
            
        change = abs(beta_current[i] - b_prev)
        rel_change = change / (abs(b_prev) + 1e-2)
        
        beta_rel_changes.append(rel_change)
        beta_converged_iter.append(rel_change < beta_tol)
        beta_satisfied_iter.append(beta_current[i] >= beta_target_conv * 0.99)
    
    all_beta_converged = all(beta_converged_iter) if beta_converged_iter else True
    all_beta_satisfied_amv = all(beta_satisfied_iter) if beta_satisfied_iter else False
    
    reliability_satisfied_final = all_beta_satisfied_amv
    if mcs_beta is not None:
        mcs_satisfied_iter = [b_val >= beta_target_conv for b_val in mcs_beta]
        reliability_satisfied_final = all(mcs_satisfied_iter)
        print(f"Checking convergence with MCS beta: {np.round(mcs_beta,5)}, Satisfied: {reliability_satisfied_final}")
    else:
        print(f"Checking convergence with AMV beta: {np.round(beta_current,5)}, Satisfied (AMV): {all_beta_satisfied_amv}")

    converged = design_converged and obj_converged and all_beta_converged
    
    status = {
        'design_change_rel': design_change_rel, 'design_converged': design_converged,
        'obj_rel_change': obj_change_rel, 'obj_converged': obj_converged,
        'beta_rel_changes': beta_rel_changes, 'beta_converged_iter': beta_converged_iter,
        'beta_satisfied_amv': beta_satisfied_iter, 'all_beta_converged': all_beta_converged,
        'all_beta_satisfied_amv': all_beta_satisfied_amv,
        'mcs_beta_values_for_check': mcs_beta,
        'final_reliability_satisfied_for_convergence_decision': reliability_satisfied_final
    }
    return converged, reliability_satisfied_final, status

def calculate_reliability_quality_score(beta_values, beta_target, objective_value, 
                                      w_violation=0.4, w_gap=0.3, w_count=0.2, w_obj=0.1):
    """
    Calculates the comprehensive quality score of the design point
    """
    beta_values = np.array(beta_values)
    
    violations = beta_values < beta_target
    violated_betas = beta_values[violations]
    satisfied_betas = beta_values[~violations]
    
    if len(violated_betas) > 0:
        worst_violation = beta_target - np.min(violated_betas)
        avg_violation_gap = np.mean(beta_target - violated_betas)
    else:
        worst_violation = 0.0
        avg_violation_gap = 0.0
    
    total_gap = avg_violation_gap
    violation_count_ratio = np.sum(violations) / len(beta_values)
    
    # Adjust objective function reference value for weight optimization problem
    obj_reference = 1.0  # Assume reference value of 1kg for weight
    obj_normalized = max(0, 1 - (objective_value - obj_reference) / obj_reference)
    
    violation_score = max(0, 1 - worst_violation / beta_target)
    gap_score = max(0, 1 - total_gap / beta_target)
    count_score = 1 - violation_count_ratio
    obj_score = max(0, obj_normalized)
    
    total_score = (w_violation * violation_score + 
                   w_gap * gap_score + 
                   w_count * count_score + 
                   w_obj * obj_score)
    
    details = {
        'worst_violation': worst_violation,
        'avg_violation_gap': avg_violation_gap, 
        'violation_count': np.sum(violations),
        'violation_count_ratio': violation_count_ratio,
        'violation_score': violation_score,
        'gap_score': gap_score,
        'count_score': count_score,
        'obj_score': obj_score,
        'total_score': total_score,
        'violated_indices': np.where(violations)[0].tolist(),
        'min_beta': np.min(beta_values),
        'satisfied_count': np.sum(~violations)
    }
    
    return total_score, details

def mcs_guided_optimization(design_point, kriging_models, bounds_mcs_opt, 
                           beta_target_mcs_opt, sigma_val_guided, 
                           obj_tol=1e-3, max_iter_mcs=5, beta_tolerance_factor=0.98,
                           previous_best_solution=None):
    """
    Improved MCS-guided optimization - uses global constraint protection and trust-region method, includes historical best solution
    """
    dim_guided = len(design_point)
    history = {
        'design_points': [design_point.copy()], 
        'objectives': [objective_function(design_point)],
        'mcs_betas': [], 
        'is_feasible': [],
        'quality_scores': [],
        'score_details': []
    }
    
    current_design = design_point.copy()
    current_obj = objective_function(current_design)
    
    candidate_solutions = []
    
    print("Starting improved MCS-guided optimization process (global constraint protection version)...")
    
    mcs_samples_guided = int(max(100000, 200 / (norm.cdf(-beta_target_mcs_opt)+1e-9)))
    mcs_samples_guided = min(mcs_samples_guided, 500000)
    print(f"MCS guided optimization using {mcs_samples_guided} MCS samples.")

    beta_values_mcs, _, is_feasible_mcs = mcs_reliability_analysis(
        current_design, kriging_models, sigma_val_guided, 
        num_samples=mcs_samples_guided, beta_target=beta_target_mcs_opt)
    
    initial_quality_score, initial_score_details = calculate_reliability_quality_score(
        beta_values_mcs, beta_target_mcs_opt, current_obj)
    
    history['mcs_betas'].append(beta_values_mcs)
    history['is_feasible'].append(is_feasible_mcs)
    history['quality_scores'].append(initial_quality_score)
    history['score_details'].append(initial_score_details)
    
    candidate_solutions.append({
        'design': current_design.copy(),
        'objective': current_obj,
        'beta_values': beta_values_mcs.copy(),
        'is_feasible': is_feasible_mcs,
        'quality_score': initial_quality_score,
        'score_details': initial_score_details,
        'iteration': 0,
        'source': 'initial'
    })
    
    # Add historical best solution to candidate list
    if previous_best_solution is not None:
        print("\nEvaluating historical best solution's performance under current model...")
        hist_beta_values, _, hist_is_feasible = mcs_reliability_analysis(
            previous_best_solution['design'], kriging_models, sigma_val_guided, 
            num_samples=mcs_samples_guided, beta_target=beta_target_mcs_opt)
        
        hist_quality_score, hist_score_details = calculate_reliability_quality_score(
            hist_beta_values, beta_target_mcs_opt, previous_best_solution['objective'])
        
        candidate_solutions.append({
            'design': previous_best_solution['design'].copy(),
            'objective': previous_best_solution['objective'],
            'beta_values': hist_beta_values.copy(),
            'is_feasible': hist_is_feasible,
            'quality_score': hist_quality_score,
            'score_details': hist_score_details,
            'iteration': -1,
            'source': 'previous_best'
        })
        
        print(f"Historical best solution: Quality score={hist_quality_score:.4f}, Obj={previous_best_solution['objective']:.4f}, "
              f"Feasible={hist_is_feasible}, Violations={hist_score_details['violation_count']}")
    
    print(f"Initial design point MCS β: {np.round(beta_values_mcs,5)}, Quality score: {initial_quality_score:.4f}")
    print(f"  Number of violated constraints: {initial_score_details['violation_count']}, Worst violation: {initial_score_details['worst_violation']:.3f}")
    
    if is_feasible_mcs:
        print("Initial design point already meets MCS reliability requirements, no further optimization needed.")
        return current_design, current_obj, beta_values_mcs, history, False
    
    # Record initial status of each constraint for protection of satisfied constraints
    initial_constraint_status = beta_values_mcs >= beta_target_mcs_opt
    
    best_found_during_search = None
    trust_region_radius = 0.1  # Initial trust region radius (relative to design space)
    
    for iteration_mcs in range(max_iter_mcs):
        print(f"\nMCS Guided Optimization - Iteration {iteration_mcs+1}/{max_iter_mcs}")
        print(f"  Current trust region radius: {trust_region_radius:.3f}")
        
        unsatisfied_constraints_indices = [i for i, b_val in enumerate(beta_values_mcs) 
                                         if b_val < beta_target_mcs_opt]
        if not unsatisfied_constraints_indices:
            print("All constraints meet strict MCS reliability requirements.")
            break
            
        print(f"Unsatisfied constraint indices: {unsatisfied_constraints_indices}")
        
        # Define penalized objective function, considering all constraints
        def penalized_objective(x_pen):
            obj = objective_function(x_pen)
            
            # Calculate MCS beta values for all constraints
            beta_all, _, _ = mcs_reliability_analysis(
                x_pen, kriging_models, sigma_val_guided,
                num_samples=max(5000, mcs_samples_guided // 10),  # Use fewer samples for speed
                beta_target=beta_target_mcs_opt)
            
            penalty = 0.0
            
            # Apply penalty to unsatisfied constraints
            for i, beta_i in enumerate(beta_all):
                if beta_i < beta_target_mcs_opt:
                    violation = beta_target_mcs_opt - beta_i
                    # Apply larger penalty to constraints that were already unsatisfied
                    if i in unsatisfied_constraints_indices:
                        penalty += 100 * violation**2
                    # Apply even larger penalty to constraints that were satisfied but are now violated (protection mechanism)
                    elif initial_constraint_status[i]:
                        penalty += 500 * violation**2
                    else:
                        penalty += 50 * violation**2
            
            # Give reward for improved constraints
            for i in unsatisfied_constraints_indices:
                if beta_all[i] > beta_values_mcs[i]:
                    improvement = beta_all[i] - beta_values_mcs[i]
                    penalty -= 10 * improvement  # Reward improvement
            
            return obj + penalty
        
        # Trust region bounds
        trust_bounds = []
        for j in range(dim_guided):
            lower = max(bounds_mcs_opt[j, 0], current_design[j] - trust_region_radius * (bounds_mcs_opt[j, 1] - bounds_mcs_opt[j, 0]))
            upper = min(bounds_mcs_opt[j, 1], current_design[j] + trust_region_radius * (bounds_mcs_opt[j, 1] - bounds_mcs_opt[j, 0]))
            trust_bounds.append((lower, upper))
        
        # Multi-start local search
        best_result = None
        best_pen_obj = float('inf')
        
        # Try multiple starting points
        start_points = [current_design]
        for _ in range(2):  # Additional random starting points
            random_start = current_design.copy()
            for j in range(dim_guided):
                random_start[j] = np.random.uniform(trust_bounds[j][0], trust_bounds[j][1])
            start_points.append(random_start)
        
        for start_point in start_points:
            try:
                result_local = minimize(
                    penalized_objective, start_point, method='L-BFGS-B',
                    bounds=trust_bounds,
                    options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
                )
                
                if result_local.fun < best_pen_obj:
                    best_pen_obj = result_local.fun
                    best_result = result_local
            except:
                continue
        
        if best_result is None:
            print("  Optimization failed, shrinking trust region radius.")
            trust_region_radius *= 0.5
            continue
        
        new_design = enforce_bounds(best_result.x, bounds_mcs_opt)
        new_obj = objective_function(new_design)
        
        # Evaluate new design point
        new_beta_values, _, new_is_feasible = mcs_reliability_analysis(
            new_design, kriging_models, sigma_val_guided, 
            num_samples=mcs_samples_guided, beta_target=beta_target_mcs_opt)
        
        new_quality_score, new_score_details = calculate_reliability_quality_score(
            new_beta_values, beta_target_mcs_opt, new_obj)
        
        print(f"Iteration {iteration_mcs+1} Candidate Result: X={np.round(new_design,5)}")
        print(f"  Obj={new_obj:.4f}, MCS β={np.round(new_beta_values,5)}")
        print(f"  Quality score={new_quality_score:.4f}, Number of violated constraints={new_score_details['violation_count']}")
        print(f"  Worst violation={new_score_details['worst_violation']:.3f}")
        
        # Decide whether to accept new design point
        accept_new_point = False
        
        # Calculate constraint improvement
        constraints_improved = 0
        constraints_worsened = 0
        for i in range(len(kriging_models)):
            if new_beta_values[i] > beta_values_mcs[i] + 0.1:
                constraints_improved += 1
            elif new_beta_values[i] < beta_values_mcs[i] - 0.1:
                constraints_worsened += 1
        
        print(f"  Constraints improved: {constraints_improved}, Constraints worsened: {constraints_worsened}")
        
        # Acceptance criteria
        if new_quality_score > initial_quality_score * 0.95:  # Quality not significantly degraded
            if constraints_improved > constraints_worsened:  # More improvement than worsening
                accept_new_point = True
                print("  Accepted new design point (more improvement than worsening)")
            elif new_quality_score > history['quality_scores'][-1]:  # Overall quality improved
                accept_new_point = True
                print("  Accepted new design point (overall quality improved)")
            elif new_score_details['worst_violation'] < history['score_details'][-1]['worst_violation'] * 0.9:
                accept_new_point = True
                print("  Accepted new design point (worst violation reduced)")
        
        if accept_new_point:
            # Update current point
            current_design = new_design.copy()
            current_obj = new_obj
            beta_values_mcs = new_beta_values.copy()
            is_feasible_mcs = new_is_feasible
            
            # If improved, can slightly increase trust region
            if new_quality_score > history['quality_scores'][-1] * 1.05:
                trust_region_radius = min(trust_region_radius * 1.2, 0.5)
                print(f"  Increased trust region radius to: {trust_region_radius:.3f}")
        else:
            print("  Rejected new design point, shrinking trust region radius")
            trust_region_radius *= 0.7
            
            # If trust region too small, try random perturbation
            if trust_region_radius < 0.01:
                print("  Trust region too small, attempting random perturbation")
                for j in range(dim_guided):
                    perturbation = np.random.uniform(-0.01, 0.01) * (bounds_mcs_opt[j, 1] - bounds_mcs_opt[j, 0])
                    current_design[j] = np.clip(current_design[j] + perturbation, 
                                              bounds_mcs_opt[j, 0], bounds_mcs_opt[j, 1])
                trust_region_radius = 0.05
        
        # Record history
        history['design_points'].append(current_design.copy())
        history['objectives'].append(current_obj)
        history['mcs_betas'].append(beta_values_mcs.copy())
        history['is_feasible'].append(is_feasible_mcs)
        history['quality_scores'].append(new_quality_score if accept_new_point else history['quality_scores'][-1])
        history['score_details'].append(new_score_details if accept_new_point else history['score_details'][-1])
        
        candidate_solutions.append({
            'design': current_design.copy(),
            'objective': current_obj,
            'beta_values': beta_values_mcs.copy(),
            'is_feasible': is_feasible_mcs,
            'quality_score': history['quality_scores'][-1],
            'score_details': history['score_details'][-1],
            'iteration': iteration_mcs + 1,
            'source': 'optimization'
        })
        
        if is_feasible_mcs:
            print("Found design point that meets MCS reliability requirements.")
            best_found_during_search = candidate_solutions[-1]
            break
        
        # Early stopping condition
        if trust_region_radius < 0.001:
            print("Trust region radius too small, stopping optimization.")
            break
    
    # Smart selection of final solution (improved: includes historical best solution)
    print(f"\n=== Smart fallback mechanism analysis (including historical best) ===")
    print(f"Number of candidate solutions: {len(candidate_solutions)}")
    
    # Separate historical solutions and current iteration solutions
    historical_solutions = [sol for sol in candidate_solutions if sol['source'] == 'previous_best']
    current_iter_solutions = [sol for sol in candidate_solutions if sol['source'] != 'previous_best']
    
    if historical_solutions:
        hist_sol = historical_solutions[0]
        print(f"Historical best solution: Quality score={hist_sol['quality_score']:.4f}, "
              f"Obj={hist_sol['objective']:.4f}, Feasible={hist_sol['is_feasible']}")
    
    feasible_candidates = [sol for sol in candidate_solutions if sol['is_feasible']]
    if feasible_candidates:
        print(f"Found {len(feasible_candidates)} strictly feasible solutions")
        best_feasible = min(feasible_candidates, key=lambda x: x['objective'])
        final_solution = best_feasible
        
        # If historical solution is better, prompt user
        if historical_solutions and hist_sol['is_feasible'] and hist_sol['objective'] < best_feasible['objective']:
            print(f"Note: Historical best solution has a smaller objective function value!")
    else:
        print("No strictly feasible solution found, comparing quality of all candidates")
        
        # Select the solution with the best quality
        quality_candidates = sorted(candidate_solutions, 
                                  key=lambda x: (x['quality_score'], -x['score_details']['worst_violation']), 
                                  reverse=True)
        
        print("\nCandidate Solutions Sorted (by Quality Score):")
        for i, sol in enumerate(quality_candidates[:5]):
            source_info = f"Historical Best" if sol['source'] == 'previous_best' else f"Iteration {sol['iteration']}"
            print(f"  Rank {i+1}: {source_info}, Score={sol['quality_score']:.4f}, "
                  f"Obj={sol['objective']:.1f}, Violations={sol['score_details']['violation_count']}")
            print(f"    β values={np.round(sol['beta_values'], 5)}")
        
        final_solution = quality_candidates[0]
    
    # Determine whether to update
    should_update = True
    if final_solution['source'] == 'previous_best':
        print("\nChoosing to keep historical best solution, not updating design point")
        should_update = False
    else:
        # If current solution is not better than historical, can choose not to update
        if historical_solutions and hist_sol['quality_score'] > final_solution['quality_score'] * 1.02:
            print("\nCurrent optimization result is not better than historical best, recommend keeping historical solution")
            final_solution = hist_sol
            should_update = False
    
    final_design = final_solution['design']
    final_obj = final_solution['objective'] 
    final_beta = final_solution['beta_values']
    
    final_design = enforce_bounds(final_design, bounds_mcs_opt)
    final_obj = objective_function(final_design)
    
    print(f"\nMCS Guided Optimization Final Result:")
    print(f"  Design point: {np.round(final_design,5)}")
    print(f"  Objective function: {final_obj:.4f}")  
    print(f"  MCS β: {np.round(final_beta,5)}")
    print(f"  Quality score: {final_solution['quality_score']:.4f}")
    print(f"  Source: {final_solution['source']} (Iteration {final_solution['iteration']})")
    print(f"  Recommended to update design point: {should_update}")
    
    return final_design, final_obj, final_beta, history, should_update

def analyze_candidate_solutions(candidate_solutions, beta_target):
    """
    Analyzes and compares detailed information of candidate solutions
    """
    print("\n=== Candidate Solution Detailed Analysis ===")
    
    for i, sol in enumerate(candidate_solutions):
        print(f"\nCandidate Solution {i+1} (Iteration {sol['iteration']}):")
        print(f"  Design point: {np.round(sol['design'], 5)}")
        print(f"  Objective function: {sol['objective']:.4f}")
        print(f"  β values: {np.round(sol['beta_values'], 5)}")
        print(f"  Strictly Feasible: {sol['is_feasible']}")
        print(f"  Quality Score: {sol['quality_score']:.4f}")
        
        details = sol['score_details']
        print(f"  Violated Constraints: {details['violated_indices']} (Total {details['violation_count']})")
        print(f"  Worst Violation: {details['worst_violation']:.3f}")
        print(f"  Average Violation Gap: {details['avg_violation_gap']:.3f}")
        print(f"  Minimum β value: {details['min_beta']:.3f}")
        
        violated_gaps = []
        for idx in details['violated_indices']:
            gap = beta_target - sol['beta_values'][idx]
            violated_gaps.append(f"β{idx} gap {gap:.3f}")
        if violated_gaps:
            print(f"  Specific Gaps: {', '.join(violated_gaps)}")

def plot_iteration_history(objective_history_plot, beta_history_plot, beta_target_plot, num_constraints):
    """
    Plots the iteration history
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    iterations_plot = np.arange(len(objective_history_plot))
    
    ax1.plot(iterations_plot, objective_history_plot, 'o-', color='blue', linewidth=2, markersize=6)
    ax1.set_ylabel('Objective Function Value (kg)')
    ax1.set_title('Weight Objective Function Iteration History')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    beta_history_arr = np.array(beta_history_plot)
    if beta_history_arr.ndim == 1 and num_constraints > 0 :
        beta_history_arr = beta_history_arr.reshape(1, -1)

    if beta_history_arr.shape[0] > 0 and beta_history_arr.shape[1] == num_constraints:
        for i in range(num_constraints):
            color = plt.cm.viridis(i / max(1, num_constraints -1)) if num_constraints > 1 else 'green'
            ax2.plot(iterations_plot, beta_history_arr[:, i], 'o-', 
                    label=f'Constraint {i+1} β', linewidth=1.5, markersize=5, color=color, alpha=0.8)
    
    ax2.axhline(y=beta_target_plot, color='r', linestyle='--', linewidth=2, label=f'Target β={beta_target_plot}')
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Reliability Index β (AMV)')
    ax2.set_title('AMV Reliability Index Iteration History')
    ax2.grid(True, linestyle=':', alpha=0.7)
    if num_constraints <= 10 :
      ax2.legend(loc='best', fontsize='small')
    else:
      ax2.legend(loc='best', fontsize='x-small', ncol=2)

    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(123)
    
    # --- Parameters for 3D engineering problem ---
    dim_main = 3
    num_constraints_main = 2
    
    # Set standard deviations based on problem definition: X_j ~ N(d_j, 0.025^2), X ~ N(L, 0.5^2)
    sigma_array_main = np.array([0.025, 0.025, 0.5])  # [d1_std, d2_std, L_std]
    
    beta_target_main = 2.0

    initial_design = np.array([5.5, 6.0, 300.0])  # [d1, d2, L]

    bounds_main = np.array([
        [4.5, 5.5],    # d1 bounds
        [5.6, 6.5],    # d2 bounds  
        [200.0, 300.0] # L bounds
    ])
    
    # Algorithm parameters - adjusted for 3D problem
    max_total_samples_main = 20 * dim_main  # Max samples 60
    max_iter_main = 5  # Max iterations
    num_samples_per_iter_main = max(2, dim_main // 2)  # Number of new samples per iteration

    gpy_restarts = 5
    # --- End of Parameters ---

    print("=== 3D Engineering Reliability Optimization Problem ===")
    print(f"Design Variables: d1 ∈ [{bounds_main[0,0]}, {bounds_main[0,1]}], d2 ∈ [{bounds_main[1,0]}, {bounds_main[1,1]}], L ∈ [{bounds_main[2,0]}, {bounds_main[2,1]}]")
    print(f"Constraints: g1 = P_outlet - 30000 ≥ 0, g2 = 0.015 + M_outlet ≥ 0")
    print(f"Objective: Minimize Weight")
    print(f"Reliability Target: β_target = {beta_target_main}")
    print(f"Uncertainty: d1,d2 ~ N(μ, 0.025²), L ~ N(μ, 0.5²)")

    # Initialize with predefined data
    print("\n=== Initialization Stage ===")
    X_initial, Y_initial = initialize_with_existing_data()
    
    # Calculate constraint function values
    print("Calculating constraint function values for initial samples...")
    Y_constraints = batch_constraint_evaluation(X_initial)

    print(f"Number of initial sample points: {len(X_initial)}")
    print(f"Constraint function dimension: {Y_constraints.shape}")
    
    # Build initial Kriging models
    print("Building initial Kriging models...")
    kriging_models = []
    for i in range(num_constraints_main):
        kernel = GPy.kern.RBF(input_dim=dim_main, variance=1.0, lengthscale=0.5 * np.ones(dim_main), ARD=True)
        kernel.variance.constrain_bounded(1e-3, 1e3)
        kernel.lengthscale.constrain_bounded(0.01, 10.0)

        model = GPy.models.GPRegression(X_initial, Y_constraints[:, i].reshape(-1, 1), kernel, normalizer=True)
        model.likelihood.variance.constrain_bounded(1e-8, 0.1)
        model.name = f"Constraint_{i+1}"
        
        try:
            model.optimize_restarts(num_restarts=gpy_restarts, optimizer='lbfgsb', robust=True, messages=False, verbose=False)
        except Exception as e_opt:
            print(f"Warning: GPy optimization for model {i+1} failed: {e_opt}. Model may not be optimized.")
        kriging_models.append(model)
        print(f"Model {model.name} initialization complete.")

    # Initialize history records
    design_history = [initial_design.copy()]
    objective_history = [objective_function(initial_design)]
    beta_history_amv = []
    
    X_all_samples = X_initial.copy()
    Y_all_samples_constraints = Y_constraints.copy()

    design_point = initial_design.copy()
    previous_design_for_conv = initial_design.copy()
    previous_objective_for_conv = objective_function(initial_design)
    previous_beta_amv_for_conv = None

    # Add global best solution tracking
    global_best_solution = {
        'design': initial_design.copy(),
        'objective': objective_function(initial_design),
        'mcs_beta': None,
        'quality_score': 0.0,
        'iteration': 0,
        'is_feasible': False,
        'amv_beta': None
    }
    
    # Add consecutive no improvement counter
    no_improvement_count = 0

    print(f"\n=== Starting Main Optimization Loop ===")
    print(f"Initial design point: {np.round(initial_design, 5)}")
    print(f"Initial weight: {objective_function(initial_design):.4f} kg")

    for iteration_main in range(max_iter_main):
        print(f"\nMain Iteration: {iteration_main+1}/{max_iter_main}, Current Design Point: {np.round(design_point,5)}, Weight: {objective_function(design_point):.4f} kg")
        
        current_amv_beta_values = []
        current_mpp_points = []
        print("  Calculating AMV reliability index for current design point...")
        for i, model in enumerate(kriging_models):
            mpp, beta_val_mpp = find_mpp_improved(model, design_point, sigma_array_main, beta_target_main)
            current_mpp_points.append(mpp)
            current_amv_beta_values.append(beta_val_mpp)
        current_amv_beta_values = np.array(current_amv_beta_values)
        beta_history_amv.append(current_amv_beta_values.copy())
        
        if iteration_main > 0:
            converged, reliability_satisfied_conv, conv_status = check_robust_convergence(
                design_point, previous_design_for_conv,
                objective_function(design_point), previous_objective_for_conv,
                current_amv_beta_values, previous_beta_amv_for_conv,
                beta_target_main, 
                mcs_beta=None
            )
            print(f"  Convergence Check: Design point relative change={conv_status['design_change_rel']:.2e} (Tol:{1e-3}), "
                  f"Objective function relative change={conv_status['obj_rel_change']:.2e} (Tol:{1e-3}), "
                  f"AMV β relative change median={np.median(conv_status['beta_rel_changes']):.2e} (Tol:{0.05})")
            print(f"  Overall Convergence: {converged}, AMV Reliability Satisfied: {conv_status['all_beta_satisfied_amv']}")

            if converged:
                print("  Design has converged. Performing final reliability check...")
                final_check_mcs_beta, _, final_check_mcs_feasible = mcs_reliability_analysis(
                    design_point, kriging_models, sigma_array_main, num_samples=100000, beta_target=beta_target_main)
                print(f"  High-fidelity MCS β for converged point ({np.round(design_point,5)}): {np.round(final_check_mcs_beta,3)}, Is feasible: {final_check_mcs_feasible}")
                if final_check_mcs_feasible:
                    print("  Design converged and satisfies MCS reliability requirements. Optimization finished.")
                    break
                else:
                    print("  Design converged but did not meet final MCS reliability. Attempting MCS guided optimization...")
                    pass

        previous_design_for_conv = design_point.copy()
        previous_objective_for_conv = objective_function(design_point)
        previous_beta_amv_for_conv = current_amv_beta_values.copy()

        feasibility_status = check_feasibility(current_amv_beta_values, beta_target_main, kriging_models, design_point, current_mpp_points)
        print(f"  AMV Feasibility Status: {feasibility_status}")
        
        # Select new sample points
        if X_all_samples.shape[0] < max_total_samples_main:
            print("  Selecting new sample points...")
            new_samples = select_new_samples(
                feasibility_status, kriging_models, bounds_main, design_point, 
                beta_target_main, sigma_array_main[0],  # Pass first std dev as reference
                num_samples_per_iter=num_samples_per_iter_main, 
                max_total_samples=max_total_samples_main
            )
            
            if len(new_samples) > 0:
                print(f"  Added {len(new_samples)} new sample points. Need to get simulation results...")
                new_Y_constraints = batch_constraint_evaluation(new_samples)
                
                X_all_samples = np.vstack((X_all_samples, new_samples))
                Y_all_samples_constraints = np.vstack((Y_all_samples_constraints, new_Y_constraints))
                
                print("  Updating Kriging models...")
                for i_m, model_m in enumerate(kriging_models):
                    model_m.set_XY(X_all_samples, Y_all_samples_constraints[:, i_m].reshape(-1, 1))
                    try:
                        model_m.optimize_restarts(num_restarts=gpy_restarts, optimizer='lbfgsb', robust=True, messages=False, verbose=False)
                    except Exception as e_opt_update:
                         print(f"Warning: GPy update for model {model_m.name} failed: {e_opt_update}.")
                print(f"  Model update complete. Total samples: {X_all_samples.shape[0]}")
            else:
                print("  No new samples added this round (perhaps max limit reached or no suitable candidates).")
        else:
            print("  Maximum number of samples reached, no new samples will be added.")

        print("  Starting two-stage optimization (finding next design point)...")
        new_design_point, is_feasible_stage2_mcs = two_stage_optimization(
            design_point, kriging_models, bounds_main, beta_target_main, 
            sigma_array_main, sigma_array_main, use_mcs=True)
        
        print(f"  Two-stage optimization complete. New candidate design point: {np.round(new_design_point,5)}, Weight: {objective_function(new_design_point):.4f} kg, MCS Feasible (rough): {is_feasible_stage2_mcs}")

        print("  Performing high-fidelity MCS evaluation on new candidate design point...")
        mcs_beta_new_design, _, mcs_feasible_new_design = mcs_reliability_analysis(
            new_design_point, kriging_models, sigma_array_main, num_samples=100000, beta_target=beta_target_main)
        print(f"  High-fidelity MCS β for new candidate point ({np.round(new_design_point,5)}): {np.round(mcs_beta_new_design,3)}, Is feasible: {mcs_feasible_new_design}")

        if not mcs_feasible_new_design:
            if iteration_main < max_iter_main - 1:
                print("  New candidate design point does not satisfy high-fidelity MCS reliability. Starting MCS guided optimization...")
                # Pass global best solution
                final_design_guided, final_obj_guided, final_mcs_beta_guided, mcs_history, should_update = mcs_guided_optimization(
                    new_design_point, kriging_models, bounds_main, beta_target_main, sigma_array_main,
                    previous_best_solution=global_best_solution if global_best_solution['mcs_beta'] is not None else None
                )
                
                if should_update:
                    new_design_point = final_design_guided
                    print(f"  MCS guided optimization found a better design point: {np.round(new_design_point,5)}")
                    print(f"    Weight: {final_obj_guided:.4f} kg, MCS β: {np.round(final_mcs_beta_guided,3)}")
                else:
                    # If not updating, it might be because the historical solution is better
                    if global_best_solution['mcs_beta'] is not None:
                        print(f"  Retaining historical best solution: Quality score={global_best_solution['quality_score']:.4f}")
                        new_design_point = global_best_solution['design'].copy()
            else:
                print("  Last main iteration, skipping MCS guided optimization.")

        # Evaluate and update global best solution
        print("\n  Evaluating current iteration results...")
        current_mcs_beta, _, current_is_feasible = mcs_reliability_analysis(
            new_design_point, kriging_models, sigma_array_main, 
            num_samples=100000, beta_target=beta_target_main)
        
        current_quality_score, current_score_details = calculate_reliability_quality_score(
            current_mcs_beta, beta_target_main, objective_function(new_design_point))
        
        print(f"  Current solution: Quality score={current_quality_score:.4f}, Feasible={current_is_feasible}, "
              f"Violations={current_score_details['violation_count']}")
        
        # Logic for updating global best solution
        update_global_best = False
        
        if current_is_feasible:
            if not global_best_solution['is_feasible']:
                # First feasible solution
                update_global_best = True
                print(f"  *** Found first feasible solution! ***")
            elif objective_function(new_design_point) < global_best_solution['objective']:
                # Better feasible solution
                update_global_best = True
                print(f"  *** Found a better feasible solution! Weight reduction: {global_best_solution['objective'] - objective_function(new_design_point):.4f} kg ***")
        elif not global_best_solution['is_feasible'] and current_quality_score > global_best_solution['quality_score']:
            # If no feasible solution yet, select the one with higher quality score
            update_global_best = True
            print(f"  *** Found a higher quality solution! Score improvement: {current_quality_score - global_best_solution['quality_score']:.4f} ***")
        
        if update_global_best:
            global_best_solution = {
                'design': new_design_point.copy(),
                'objective': objective_function(new_design_point),
                'mcs_beta': current_mcs_beta.copy(),
                'quality_score': current_quality_score,
                'iteration': iteration_main + 1,
                'is_feasible': current_is_feasible,
                'amv_beta': current_amv_beta_values.copy()
            }
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"  Consecutive no improvement count: {no_improvement_count}")
        
        # Early stopping condition check
        if global_best_solution['is_feasible'] and iteration_main >= 2:
            # Condition 1: Found feasible solution and no improvement for several consecutive iterations
            if no_improvement_count >= 2:
                print(f"\n*** Early stopping triggered: Feasible solution found and no improvement for {no_improvement_count} consecutive iterations ***")
                print(f"    Best solution from iteration {global_best_solution['iteration']}")
                design_point = global_best_solution['design'].copy()
                break
            
            # Condition 2: Current solution quality significantly dropped
            if current_quality_score < global_best_solution['quality_score'] * 0.9:
                print(f"\n*** Early stopping triggered: Current solution quality significantly dropped ({current_quality_score:.4f} < {global_best_solution['quality_score'] * 0.9:.4f}) ***")
                design_point = global_best_solution['design'].copy()
                break
        
        # Condition 3: If quality score is very high (close to 1.0) and reliability requirement is met
        if current_quality_score > 0.995 and current_is_feasible:
            print(f"\n*** Early stopping triggered: Found high-quality feasible solution (Score={current_quality_score:.4f}) ***")
            break
        
        # Update design point
        design_point = new_design_point.copy()
        design_history.append(design_point.copy())
        objective_history.append(objective_function(design_point))
        
        if X_all_samples.shape[0] >= max_total_samples_main and iteration_main > dim_main:
            if abs(objective_history[-1] - objective_history[-2]) < 1e-3 * abs(objective_history[-2]):
                print("  Maximum sample limit reached and objective function improvement is minimal, considering early termination.")

    # --- Final Evaluation ---
    print("\n=== Main Optimization Loop Ended. ===")
    
    # Ensure using the global best solution
    if global_best_solution['mcs_beta'] is not None:
        print(f"\n=== Final result uses Global Best Solution (from iteration {global_best_solution['iteration']}) ===")
        final_design_point = global_best_solution['design']
        final_objective = global_best_solution['objective']
        final_mcs_betas = global_best_solution['mcs_beta']
        final_mcs_is_feasible = global_best_solution['is_feasible']
    else:
        # If no global best solution was recorded, use the last design point
        final_design_point = design_history[-1]
        final_objective = objective_history[-1]
        final_mcs_betas, final_mcs_pf_values, final_mcs_is_feasible = mcs_reliability_analysis(
            final_design_point, kriging_models, sigma_array_main, num_samples=200000, beta_target=beta_target_main)

    print("Calculating final AMV and MCS reliability indices...")
    final_amv_betas = []
    for model in kriging_models:
        _, beta_amv = find_mpp_improved(model, final_design_point, sigma_array_main, beta_target_main)
        final_amv_betas.append(beta_amv)
    final_amv_betas = np.array(final_amv_betas)

    # Calculate final probability of failure
    final_mcs_pf_values = norm.cdf(-final_mcs_betas)

    print("\n=== 3D Engineering Optimization Results Summary ===")
    print(f"Final Optimized Design Point: d1={final_design_point[0]:.4f}, d2={final_design_point[1]:.4f}, L={final_design_point[2]:.2f}")
    print(f"Final Weight: {final_objective:.4f} kg")
    print(f"Final AMV Method Reliability Index: β1={final_amv_betas[0]:.3f}, β2={final_amv_betas[1]:.3f}")
    print(f"Final High-Fidelity MCS Method Reliability Index: β1={final_mcs_betas[0]:.3f}, β2={final_mcs_betas[1]:.3f}")
    print(f"Final High-Fidelity MCS Method Probability of Failure: Pf1={final_mcs_pf_values[0]:.2e}, Pf2={final_mcs_pf_values[1]:.2e}")
    print(f"Final Design Point satisfies MCS Reliability Requirement (β_target={beta_target_main}): {final_mcs_is_feasible}")
    print(f"Total number of samples used: {X_all_samples.shape[0]}")
    
    # Output constraint function values
    final_constraint_values = constraint_functions_interactive(final_design_point)
    print(f"Final Constraint Function Values: g1(P_outlet)={final_constraint_values[0]:.2f} Pa, g2(0.02+M_outlet)={final_constraint_values[1]:.6f}")
    
    if objective_history and beta_history_amv:
        if len(objective_history) == len(beta_history_amv) + 1:
             plot_iteration_history(objective_history[1:], beta_history_amv, beta_target_main, num_constraints_main)
        elif len(objective_history) == len(beta_history_amv):
             plot_iteration_history(objective_history, beta_history_amv, beta_target_main, num_constraints_main)
        else:
            print(f"History lengths do not match, cannot plot iteration history. Obj len: {len(objective_history)}, Beta AMV len: {len(beta_history_amv)}")

    return final_design_point, final_objective, final_mcs_betas, final_mcs_pf_values, kriging_models, bounds_main

if __name__ == '__main__':
    start_time = time.time()
    main_results = main()
    end_time = time.time()
    print(f"\nTotal computation time: {(end_time - start_time)/60:.2f} minutes")
    
    print("\n=== Instructions for Use ===")
    print("1. The program is initialized with 30 sets of predefined experimental data.")
    print("2. When new simulation results are needed, the program will prompt for input.")
    print("3. Please enter P_outlet (Pa) and M_outlet (kg/s) as prompted.")
    print("4. The program will automatically save all entered data to avoid redundant calculations.")
    print("5. You can interrupt the program at any time (Ctrl+C); entered data will not be lost.")