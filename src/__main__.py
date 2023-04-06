#! /usr/bin/env python3
import argparse
from experiments.interpolated import interpolated_experiment
from experiments.shaping import shaping_experiment

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--experiment', type=str, default='interpolated',
                      help='Which experiment to run (one of "interpolated" or "shaping"))')
  parser.add_argument('-r', '--results_path', type=str, default='results.json',
                      help='Path to save results')
  args = parser.parse_args()
  if args.experiment == 'interpolated':
    interpolated_experiment(args.results_path)
  elif args.experiment == 'shaping':
    shaping_experiment(args.results_path)
  else:
    raise ValueError(f"Invalid experiment {args.experiment}")
