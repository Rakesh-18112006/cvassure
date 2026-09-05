# cvassure — everything runs offline.
PY := .venv/bin/python
CV := .venv/bin/cvassure

.PHONY: help bootstrap verify demo reproduce results clean test lint

help:
	@echo "make bootstrap  build the local dataset, models and keys (once)"
	@echo "make verify     run every test plus the offline assertion"
	@echo "make demo       the four-minute judge sequence"
	@echo "make reproduce  regenerate every number and figure from scratch, offline"
	@echo "make clean      remove generated results"

bootstrap:
	$(PY) -c "from cvassure.datasets import synth; \
	  synth.build('data/synth10', n_per_class=30, n_classes=10, n_contributors=5, seed=0); \
	  synth.build_ood_pool('data/ood_pool', n=80, seed=99)"
	$(PY) -c "import numpy as np; from cvassure.datasets import synth, toy_model; \
	  x=np.stack([np.asarray(synth.make_image(c,c*1000+k),dtype=np.float32).transpose(2,0,1)/255. \
	    for c in range(10) for k in range(30)]); \
	  y=np.asarray([c for c in range(10) for k in range(30)]); \
	  toy_model.build_and_export('models', x, y, name='vendor', seed=1, epochs=12); \
	  toy_model.build_and_export('models', x, y, name='impostor', seed=77, epochs=12)"
	test -f keys/priv.pem || $(CV) provenance init-keys --out keys/
	@echo "Bootstrap complete."

test:
	$(PY) -m pytest -q

verify: test
	@echo "Checking that nothing reaches the network..."
	$(PY) -m cvassure.cli --offline-assert ingest inspect \
	  --dataset data/synth10 --model models/vendor.onnx --access-tier 1 > /dev/null
	@echo "PASS  the audit ran with the network layer disabled."

demo:
	$(CV) demo --workdir results/demo

results:
	$(PY) experiments/run_all.py --out results
	$(PY) experiments/aggregate.py --out results

reproduce: clean bootstrap test results
	@echo
	@echo "Everything regenerated. Tables and figures are in results/."

clean:
	rm -rf results/tables results/plots results/RESULTS.md results/COVERAGE.md \
	       results/sweep_raw.jsonl results/sweep.json results/.sweep
