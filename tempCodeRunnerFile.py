    elif model_name == 'MVK':
        print(f"**Running Mars-van Krevelen Model (2 parameters: kr, ko)**")
        # Initial guess and bounds for MvK
        par0 = np.array([0.1, 15.0]) 
        
        # --- MODIFIED ---
        # Give the optimizer more freedom. 1e-6 is (0.000001)
        parlb = np.array([1e-6, 1e-6]) 
        parub = np.array([1000.0, 1000.0]) # Let it search up to 1000
        # --- END MODIFIED ---

        kinetic_model = mars_van_krevelen