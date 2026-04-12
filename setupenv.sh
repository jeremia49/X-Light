source .venv/bin/activate
export SUMO_HOME=''
export $(uv run python -c "import sumo; print('SUMO_HOME=' + sumo.SUMO_HOME)")
export PYTHONPATH=''
export PYTHONPATH=${PYTHONPATH}:$(pwd)
