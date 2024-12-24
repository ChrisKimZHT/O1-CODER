CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename taco_test \
    --api OpenAI \
    --model_ckpt qwen2-instruct \
    --note default \
    --num_rollouts 12 \
    --verbose \
    --max_depth_allowed 10