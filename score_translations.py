import argparse
import subprocess


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-file', type=str, default='data/sources.txt')
    parser.add_argument('-t', '--translation-file', type=str, default='data/candidates_N20_t0.9_p0.6.txt')
    parser.add_argument('-m', '--metric', type=str, default='COMET')
    parser.add_argument('-d', '--decoding-type', type=str, default='MBR')
    args = parser.parse_args()

    # Score translations
    output_file = f'{args.translation_file.replace("candidates", "scores")[:-4]}_{args.metric.lower()}_{args.decoding_type.lower()}.txt'
    num_candidates = int(args.translation_file[args.translation_file.find('N'):].split('_')[0][1:])
    
    # Using COMET
    if args.metric == 'COMET':
        
        # Using MBR
        if args.decoding_type == 'MBR':
            # raise NotImplementedError
            subprocess.run(
                f'comet-mbr -s {args.source_file} -t {args.translation_file} --num_sample {num_candidates} -o {output_file}', 
                shell=True, check=True, capture_output=True, text=True
            )
        
        # Using N-best reranking
        elif args.decoding_type == 'NBest':
            raise NotImplementedError
    
    # Using other metrics (BLEU, chrF...)
    else:
        raise NotImplementedError

        
if __name__ == '__main__':
    main()