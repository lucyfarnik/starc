#! /usr/bin/env python3
import argparse
from experiments import interpolated_experiment, shaping_experiment
from experiments import handpicked_experiment, discount_changes_experiment

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--experiment', type=str, default='interpolated',
                      choices=['interpolated', 'shaping', 'handpicked', 'discount_changes'],
                      help='Which experiment to run')
  parser.add_argument('-r', '--results_path', type=str, default='results.json',
                      help='Path to save results')
  args = parser.parse_args()
  if args.experiment == 'interpolated':
    interpolated_experiment(args.results_path)
  elif args.experiment == 'shaping':
    shaping_experiment(args.results_path)
  elif args.experiment == 'handpicked':
    handpicked_experiment(args.results_path)
  elif args.experiment == 'discount_changes':
    discount_changes_experiment(args.results_path)
  else:
    raise ValueError(f"Invalid experiment {args.experiment}")
