WANDB_API_KEY=b3fb3695850f3bdebfee750ead3ae8230c14ea07
DF=Dockerfile
RUN_FILE="jsrl-CORL/algorithms/finetune/ray_trainer.py"
CPUS=3

run:
	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--rm \
	jsrl-corl python $(RUN_FILE)

build:
	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

build_and_run_lunar:
	yes | sudo docker container prune

	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	$(DETACH) \
	--cpus $(CPUS) \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/jsrl-CORL" \
	jsrl-corl python $(RUN_FILE) --env LunarLander-v2 --guide_heuristic_fn lunar_lander --offline_iterations 0 --env_config '{"continuous": True}' --eval_freq 10000 --beta 10 --iql_tau 0.9 --horizon_fn goal_dist --name IQL-test --device cpu

build_and_run_antmaze:
	yes | sudo docker container prune

	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.
	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	$(DETACH) \
	--cpus $(CPUS) \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/jsrl-CORL" \
	jsrl-corl python $(RUN_FILE) --horizon_fn goal_dist --heuristic_fn perfect_heuristic --env LunarLander-v2 --normalize True --normalize_reward True --iql_deterministic False --beta 10 --buffer_size 10000000 --iql_tau 0.9 --device cpu --name variance-test --group variance_tests --eval_freq 10000 ;