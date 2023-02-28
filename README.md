# Reward distance
The idea of these experiments is to compare different EPIC-like
reward function distances and see how well they correlate with each other,
as well as with regret from rolling out the policy.

## Intended audience
This repo is meant for my co-authors.

## How to navigate this clusterfuck of a repo
As you may have guessed from, like, everything, this is a work in progress.
All the code for running the experiments is directly in the root directory.
The `tests` directory, unsurprisingly, contains racoons that will
bite you if you stare at them for too long (and maybe some tests too idk).
Directories that start with __ (double underscore) are for storing ancient
scrolls that were useful once upon a time, and when the ravens return to
Erebor they may be useful once again which is why we keep them around.

There's analysis and plotting stuff in `/analysis` and `/charts`.

## How to run
Run `python __main__.py` to run the experiments, for tests run `pytest`.
To show the charts, run the python files in the `charts` directory
(you might have to modify the path to the results file).

## Help
If you have any questions, please get in touch with me. If something in this
repo looks stupid, that's probably because it is.
