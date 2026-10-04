.PHONY: smoke benchmark browser servo lqr ppo train-ppo-v3

MODEL ?= models/ppo_v3/policy_accepted.pt

smoke:
	python tests/smoke_core.py

benchmark:
	python software/python/benchmark_suite.py --controllers lqr --fail-on-reject

browser:
	cd software/python && python active_vibration_rig_web.py

servo:
	cd software/python && python active_vibration_rig_2d.py --headless 5 --controller safe_servo --trajectory aggressive

lqr:
	cd software/python && python active_vibration_rig_2d.py --headless 5 --controller lqr --trajectory aggressive

ppo:
	@test -f "$(MODEL)" || (echo "Missing $(MODEL). Train/download a v3 checkpoint first." && exit 1)
	cd software/python && python active_vibration_rig_web.py --ppo-model ../../$(MODEL) --controller ppo --trajectory aggressive

train-ppo-v3:
	cd software/python && python pretrain_v3_damping.py --out ../../models/ppo_v3/heuristic_warmstart.pt
	cd software/python && python train_ppo_v3.py --init ../../models/ppo_v3/heuristic_warmstart.pt --steps 8192 --outdir ../../models/ppo_v3
