.PHONY: all do generate_data_statistics run_fixed_parameter_models generate_figures_eps generate_figures_png generate_results_tables clean
.DEFAULT  : all

all: generate_data_statistics \
	run_fixed_parameter_models \
	generate_figures_eps \
	generate_figures_png \
 	generate_results_tables \
	clean


generate_data_statistics:
	@echo "Running experiments..."
	python3 ./tables/mkSummary.py --save ./tables/latex/data_summary.tex


run_fixed_parameter_models:
	# Run experiments for all fixed parameter settings
	@echo "--- Parameter setting 1/3 ---"
	python3 ./main.py --params '{ \
		"horizon": 30, \
		"data_frequency": 5, \
		"x0": 0, \
		"dx": 0.1, \
		"g": 0.8, \
		"h": 0.2, \
		"dt": 1.0, \
		"meal_duration": \
		30, \
		"savgol_poly": 1, \
		"savgol_len": 15, \
		"calibration_time": 3, \
		"std_mult": 1, \
		"prediction_period": 60 \
	}' \
	--save './results/fixed_parameters_std-1'
	@echo "--- Parameter setting 2/3 ---"
	python3 ./main.py --params '{ \
		"horizon": 30, \
		"data_frequency": 5, \
		"x0": 0, \
		"dx": 0.1, \
		"g": 0.8, \
		"h": 0.2, \
		"dt": 1.0, \
		"meal_duration": \
		30, \
		"savgol_poly": 1, \
		"savgol_len": 15, \
		"calibration_time": 3, \
		"std_mult": 2, \
		"prediction_period": 60 \
	}' \
	--save './results/fixed_parameters_std-2'
	@echo "--- Parameter setting 3/3 ---"
	python3 ./main.py --params '{ \
		"horizon": 30, \
		"data_frequency": 5, \
		"x0": 0, \
		"dx": 0.1, \
		"g": 0.8, \
		"h": 0.2, \
		"dt": 1.0, \
		"meal_duration": \
		30, \
		"savgol_poly": 1, \
		"savgol_len": 15, \
		"calibration_time": 3, \
		"std_mult": 1, \
		"prediction_period": 60 \
	}' \
	--save './results/fixed_parameters_std-3'


generate_figures_eps:
	@echo "Generating EPS Figures..."
	# Background
	python3 ./figures/figure_various_meal_impacts.py \
					--save ./figures/eps/meal_responses.eps
	# Methods
	python3 ./figures/figure_glucose_compartments.py \
					--save ./figures/eps/glucose_compartments.eps
	python3 ./figures/figure_insulin_compartments.py \
					--save ./figures/eps/insulin_compartments.eps
	python3 ./figures/figure_cgm_smoothing.py \
					--save ./figures/eps/cgm-upsample-smoothing.eps
	# Results
	python3 ./figures/figure_results_compartments-explanation-1.py \
					--save ./figures/eps/figure-results-explain-1.eps
	python3 ./figures/figure_results_compartments-explanation-2.py \
					--save ./figures/eps/figure_all-compartments-prediction.eps
	python3 ./figures/figure_results_compartments-explanation-3.py \
					--save ./figures/eps/figure-results-meal-detection.eps
	python3 ./figures/figure_err_acc_jointplot.py \
					--save ./figures/eps/err-scores-correlation.eps


generate_figures_png:
	@echo "Generating PNG Figures..."
	# Background
	python3 ./figures/figure_various_meal_impacts.py \
					--save ./figures/png/meal_responses.png
	# Methods
	python3 ./figures/figure_glucose_compartments.py \
					--save ./figures/png/glucose_compartments.png
	python3 ./figures/figure_insulin_compartments.py \
					--save ./figures/png/insulin_compartments.png
	python3 ./figures/figure_cgm_smoothing.py \
					--save ./figures/png/cgm-upsample-smoothing.png
	# Results
	python3 ./figures/figure_results_compartments-explanation-1.py \
					--save ./figures/png/figure-results-explain-1.png
	python3 ./figures/figure_results_compartments-explanation-2.py \
					--save ./figures/png/figure_all-compartments-prediction.png
	python3 ./figures/figure_results_compartments-explanation-3.py \
					--save ./figures/png/figure-results-meal-detection.png
	python3 ./figures/figure_err_acc_jointplot.py \
					--save ./figures/png/err-scores-correlation.png
	# Discussion
	python3 ./figures/figure_gmaps.py \
					-s1 ./figures/png/gmaps_im.png \
					-s2 ./figures/png/gmaps_plt.png


generate_results_tables:
	@echo "Generating LaTeX Result Tables..."
	# Individual and aggregate results tables
	python3 ./results/aggregate_table.py


clean:
	@echo "Cleaning up..."
	@echo "--- ALL SET! ---"
