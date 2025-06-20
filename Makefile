install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git config pull.rebase false
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential


push-hub:
	huggingface-cli upload NazarBai/Drug-Classification ./App --repo-type=space
	huggingface-cli upload NazarBai/Drug-Classification ./Model /Model --repo-type=space
	huggingface-cli upload NazarBai/Drug-Classification ./Results /Metrics --repo-type=space

deploy: hf-login push-hub
