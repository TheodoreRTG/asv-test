# ASV Tensorflow

- Populate your commit_list:
 	- To output the latest tensorflow binary and commit number to your commit_list : Run `gen_commit_list.sh`.
	- To output a range of tensorflow binaries and commit numbers to your commit_list : Run `extract_builds.py` , there are two arguments which determine the first and last ID to crawl the CI page and save as `commit_list`.

- Run `get_commit_list.sh` it will create a `builds/{commit}` directory with all Tensorflow versions.
- Run `gen_asv_commit_list.sh`, it will create a `asv_commit_list` file with which you can run ASV with `asv run HASHFILE:asv_commit_list`.
- Make sure to update the submodules to get `inference`, then move to that directory and run:
	- `git checkout r1.1` - This _should_ be unnecessary as .gitmodules is already set to checkout `r1.1`.
	- `git cherry-pick -n 215c057fc6690a47f3f66c72c076a8f73d66cb12`
	- `git submodule update --init --recursive`
	- `cd loadgen`
	- `python setup.py bdist_wheel`
	- `cd ../vision/classification_and_detection/`
	- `python setup.py bdist_wheel`
- This should be enough to create wheels under each `./dist/` subdirectories.
