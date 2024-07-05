WANDB_API_KEY=b3fb3695850f3bdebfee750ead3ae8230c14ea07
DF=Dockerfile
RUN_FILE="GuidedRollBackRL/algorithms/finetune/ray_trainer.py"
CPUS=5

run:
	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--rm \
	grbrl python $(RUN_FILE)

build:
	sudo docker build \
	-f $(DF) \
	-t grbrl \
	.

build_and_run_combination:
	yes | docker container prune

	docker build \
	-f $(DF) \
	-t grbrl \
	.

	docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/GuidedRollBackRL" \
	grbrl python $(RUN_FILE) --env CombinationLock-v0 --learner_frac -1 --online_buffer_size 64 --guide_heuristic_fn combination_lock --offline_iterations 0 --env_config '{"horizon": 10}' --tolerance 0.75 --n_episodes 250 --eval_freq 1 --batch_size 10 --beta 10 --iql_tau 0.9 --horizon_fn time_step --name IQL-test --device cpu --online_iterations 300 --seed 0 --iql_deterministic True --enable_rollback False --sample_rate 0.9 ;


	docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/GuidedRollBackRL" \
	grbrl python $(RUN_FILE) --env CombinationLock-v0 --learner_frac -1 --online_buffer_size 64 --guide_heuristic_fn combination_lock --offline_iterations 0 --env_config '{"horizon": 10}' --tolerance 0.75 --n_episodes 250 --eval_freq 1 --batch_size 10 --beta 10 --iql_tau 0.9 --horizon_fn time_step --name IQL-test --device cpu --online_iterations 300 --seed 0 --iql_deterministic True --enable_rollback True --sample_rate 0.9 ;




build_and_run_lunar:
	yes | docker container prune

	docker build \
	-f $(DF) \
	-t grbrl \
	.

	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/GuidedRollBackRL" \
	grbrl python $(RUN_FILE) --env LunarLander-v2 --variance_learn_frac 0.0 --guide_heuristic_fn lunar_lander --offline_iterations 0 --env_config '{"continuous": True}' --eval_freq 500 --beta 10 --iql_tau 0.9 --horizon_fn variance --name IQL-test --device cpu --online_iterations 1 ;

build_and_run_antmaze:
	yes | docker container prune

	docker build \
	-f $(DF) \
	-t grbrl-corl \
	.

	docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ./algorithms/finetune:/workspace/GuidedRollBackRL/algorithms/finetune \
	-v ".:/workspace/GuidedRollBackRL" \
	grbrl-corl python $(RUN_FILE) --project AdaptiveAlphaAddDim --adaptive_alpha False --add_alpha_dim True --horizon_fn time_step --checkpoints_path checkpoints --normalize True --tolerance 0.95 --normalize_reward True --iql_deterministic False --beta 10 --learner_frac 0.2 --correct_learner_action 0.9 --iql_tau 0.9 --eval_freq 1000 --n_episodes 1 --offline_iterations 0 --online_iterations 1000000 --pretrained_policy_path GuidedRollBackRL/algorithms/finetune/checkpoints/IQL-antmaze-umaze-diverse-v2-offline/checkpoint_999999.pt --env antmaze-umaze-diverse-v2 --device cpu --enable_rollback False ;


run_variance_learner:
	sudo docker run \
	-it \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/var_functions:/workspace/GuidedRollBackRL/algorithms/finetune/var_functions \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/GuidedRollBackRL" \
	grbrl python GuidedRollBackRL/algorithms/finetune/variance_learner.py
