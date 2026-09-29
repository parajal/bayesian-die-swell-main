Code to build and run BI on four types of models: "Higdon approach", "KOH approach", "pure ML", and "pure simulation" (cf. preprint die swell paper). Loaded data is from extrusion experiments "18-09-2020_T_120_L-D_20_2_Sam_693_L_H.002".

- generate_model.py :
  		    -- Loads the data and builds the PDFs conditioned on the data and the chosen priors.
		    Most importantly:
		    -- Choose the model to run via IND_MODEL parameter
		    -- Set the number "NbPred" used for discretization of shear-rate-space used for predictions
		    -- Set the number used for discretization of (shear-rate, material-parameters)-space,
		       used for inference on the simulator (needed in the Higdon approach), either
		       (i) "NbObs_per_dim_SIM", if cartesian discretization, or
		       (ii) "n_obs_x" and "n_obs_t", if LH discretization.

- run_mcmc.py:
		-- Runs the MCMC chain.
		-- Set the number of samples with the parameter "num_results".

- plots.py:
		-- post processing. Generates .pdf plots and a statistical summary in .txt and .tex format.