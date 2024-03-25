import argparse
from vllm import LLM, SamplingParams


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-file', type=str, default='data/training/repeated_sources_N20.txt')
    parser.add_argument('-m', '--model-path', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('-ml', '--model-path-local', type=str, default='/linkhome/rech/genrce01/urb81vh/.cache/huggingface/hub/[MODEL]/snapshots/6ea1d9e6cb5e8badde77ef33d61346a31e6ff1d4')
    parser.add_argument('-hp', '--hypotheses_file', type=str, default='data/training/hypotheses_N20.txt')
    parser.add_argument('-t', '--temperature', type=float, default=0.9)
    parser.add_argument('-p', '--top-p', type=float, default=0.6)
    parser.add_argument('-mt', '--max-tokens', type=float, default=1024)
    parser.add_argument('-q', '--quantization', type=str, default=None)
    args = parser.parse_args()

    # Load model
    print('==========> Loading model...')
    if '[MODEL]' in args.model_path_local:
        model_path_local = args.model_path_local.replace('[MODEL]', 'models--' + args.model_path.replace('/', '--'))
    else:
        model_path_local = args.model_path_local

    llm = LLM(model=model_path_local, max_model_len=1024, tensor_parallel_size=1)
    print('==========> Done.\n')

    # Load source data
    print('==========> Loading source data...')
    with open(args.source_file, 'r') as f:
        prompts = [line.strip().replace('\\n', '\n') for line in f]
    print('==========> Done.\n')

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        quantization=args.quantization
    )
    
    # Generate candidates
    print('==========> Generating candidates...')
    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip().replace('\n', '\\n') for o in outputs]
    generations_txt = '\n'.join(generations)
    print('==========> Done.\n')

    # Save candidates file
    print('==========> Saving hypotheses file...')
    #with open(f'{args.save_directory}/candidates_N{args.source_file.split("_")[-1][1:-4]}_t{args.temperature}_p{args.top_p}.txt', 'w') as f:
    with open(args.hypotheses_file, 'w') as f:
        f.write(generations_txt)

    print('==========> Done.')

if __name__ == '__main__':
    main()
