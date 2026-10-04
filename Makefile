.PHONY: smoke browser servo lqr ppo

MODEL ?= models/ppo_v2_reversal.pt

smoke:
	python tests/smoke_core.py

browser:
	cd software/python && python active_vibration_rig_web.py

servo:
	cd software/python && python active_vibration_rig_2d.py --headless 5 --controller servo --trajectory aggressive

lqr:
	cd software/python && python active_vibration_rig_2d.py --headless 5 --controller lqr --trajectory aggressive

ppo:
	@test -f "$(MODEL)" || (echo "Missing $(MODEL). Train/download a checkpoint first; see docs/ARTIFACTS.md" && exit 1)
	cd software/python && python active_vibration_rig_web.py --ppo-model ../../$(MODEL) --controller ppo --trajectory aggressive
