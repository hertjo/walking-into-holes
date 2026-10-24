PYTHON ?= python

TECTONIC ?= tectonic

.PHONY: all paper paper-figures hero-renders experiments cooling-reference bunny-reference compiled-bunny engineering-housing engineering-channels engineering-offset-sweep engineering-fem-audit clean

all: paper

# The paper (single document; the appendix follows the bibliography).
paper:
	cd paper && $(TECTONIC) main.tex

# Paper figures, drawn from the tracked benchmark outputs. Needs Splinx, the author's figure tool (not public yet);
# the committed PDFs in paper/figures/ are its output. prep_*.py recolour renders / recompute fields first.
FIGURE_SCRIPTS = $(sort $(wildcard paper/figure-scripts/f_*.py paper/figure-scripts/s_*.py))
paper-figures:
	cd paper/figure-scripts && for p in prep_hero prep_submission prep_internal_cooling prep_stanford_bunny prep_accessible prep_external_fem; do $(PYTHON) $$p.py || exit 1; done
	cd paper/figure-scripts && for f in $(notdir $(FIGURE_SCRIPTS)); do $(PYTHON) $$f || exit 1; done
	cp paper/figure-scripts/out/*.pdf paper/figures/

# Path-traced Bunny views used by the hero figure (Mitsuba; see figures/requirements-render.txt).
hero-renders:
	$(PYTHON) figures/render_mesh_hero.py --size 1200 --spp 64
	$(PYTHON) figures/hero_wire_details.py
	$(PYTHON) figures/hero_thermal.py
	$(PYTHON) figures/hero_budget.py

experiments:
	$(PYTHON) experiments/run_controlled_experiments.py

cooling-reference:
	$(PYTHON) benchmarks/internal_cooling/run_reference_benchmark.py

bunny-reference:
	$(PYTHON) benchmarks/stanford_bunny/run_bunny_benchmark.py

.PHONY: bunny-budget
bunny-budget:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/bunny_budget/run_study.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/bunny_budget/check_termination.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/bunny_budget/audit.py
	$(PYTHON) benchmarks/bunny_budget/analyze.py

compiled-bunny:
	$(PYTHON) benchmarks/compiled_bunny/run_compiled_benchmark.py

engineering-housing:
	$(PYTHON) benchmarks/engineering_housing/run_housing_benchmark.py

engineering-channels:
	$(PYTHON) benchmarks/engineering_housing/run_channel_benchmark.py

engineering-offset-sweep:
	$(PYTHON) benchmarks/engineering_housing/run_offset_seam_sweep.py

engineering-fem-audit:
	$(PYTHON) benchmarks/engineering_housing/run_external_fem_baseline.py

.PHONY: critical-baselines critical-figures
critical-baselines:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) benchmarks/critical_baselines/sampling.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) benchmarks/critical_baselines/source_sampling.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) benchmarks/critical_baselines/shape_tracking.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) benchmarks/critical_baselines/connectivity.py
	$(MAKE) critical-figures

critical-figures:
	$(PYTHON) benchmarks/critical_baselines/plot_results.py
	$(PYTHON) benchmarks/critical_baselines/summarize.py

clean:
	rm -rf paper/*.pdf paper/figure-scripts/out

.PHONY: application-validation application-validation-audit
application-validation:
	$(PYTHON) benchmarks/minimum_benefit/build_backends.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 $(PYTHON) benchmarks/minimum_benefit/run_application_validation.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 $(PYTHON) benchmarks/minimum_benefit/grid_reference.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/minimum_benefit/audit_and_plot.py

application-validation-audit:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/minimum_benefit/audit_and_plot.py

.PHONY: distinct-adjoint
distinct-adjoint:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/distinct_adjoint/run_benchmark.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/distinct_adjoint/audit_benchmark.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/distinct_adjoint/run_persistent.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/distinct_adjoint/plot_results.py

.PHONY: submission-study submission-study-figures
submission-study:
	$(PYTHON) benchmarks/submission_study/build_backends.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 $(PYTHON) benchmarks/submission_study/geometry.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 $(PYTHON) benchmarks/submission_study/run_study.py --stage all
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 $(PYTHON) benchmarks/submission_study/run_coarse.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 $(PYTHON) benchmarks/submission_study/run_hit_difference.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/submission_study/analyze_study.py
	$(MAKE) submission-study-figures

submission-study-figures:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 $(PYTHON) benchmarks/submission_study/render_study.py
	OPENBLAS_NUM_THREADS=1 $(PYTHON) benchmarks/submission_study/compose_study.py
