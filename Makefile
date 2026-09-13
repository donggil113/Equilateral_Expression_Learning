.PHONY: test theory experiments report paper clean

test:
	python3 -m pytest tests/ -q

theory:
	python3 experiments/exp_theory.py
	python3 experiments/exp_equivariance.py

experiments:
	./scripts/run_all_experiments.sh

report:
	python3 scripts/make_figure1.py
	python3 scripts/make_report.py
	python3 scripts/summarize.py

paper: report
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
	  bibtex main && pdflatex -interaction=nonstopmode main.tex && \
	  pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out
