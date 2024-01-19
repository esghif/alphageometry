from functools import reduce
import operator
import os
import sys

from typing import Any, Dict, List, Sequence, Tuple


def adapt_kwargs(kwargs: Dict[str, str]) -> Dict[str, str]:
    def adapt_val(val):
        if val is None:
            return None
        if isinstance(val, list):
            return tuple(val)
        return f"{val}"
    return {
        f"--{key}": adapt_val(val)
        for key, val in kwargs.items()
    }


def flatten_list_parameter_values(keyval: Tuple[str, Any]) -> Sequence[Any]:
    key, val = keyval
    if isinstance(val, (list, tuple)):
        return (key, ) + tuple(val)
    return keyval


def make_parameter_list(parameters: Dict[str, Any]) -> List[str]:
    parameters = map(flatten_list_parameter_values, parameters.items())
    parameters = reduce(operator.add, parameters, ())
    parameters = list(filter(lambda _: _ is not None, parameters))
    return parameters

MELIAD_PATH = "/meliad_lib/meliad"
sys.path.append(MELIAD_PATH)
sys.path.append("./")

from absl import app
import alphageometry

DATA = "ag_ckpt_vocab"
work_dir = os.getcwd()

DDAR_ARGS = {
    "defs_file": f"{work_dir}/defs.txt",
    "rules_file": f"{work_dir}/rules.txt",
}

BATCH_SIZE = 2
BEAM_SIZE = 2
DEPTH = 2

SEARCH_ARGS = {
    "beam_size": BEAM_SIZE,
    "search_depth": DEPTH,
}

LM_ARGS = [
    "--ckpt_path", DATA,
    "--vocab_path", f"{DATA}/geometry.757.model",
    "--gin_search_paths", f"{MELIAD_PATH}/transformer/configs", f"{work_dir}",
    "--gin_file", "base_htrans.gin",
    "--gin_file", "size/medium_150M.gin",
    "--gin_file", "options/positions_t5.gin",
    "--gin_file", "options/lr_cosine_decay.gin",
    "--gin_file", "options/seq_1024_nocache.gin",
    "--gin_file", "geometry_150M_generate.gin",
    "--gin_param", "DecoderOnlyLanguageModelGenerate.output_token_losses=True",
    "--gin_param", f"TransformerTaskConfig.batch_size={BATCH_SIZE}",
    "--gin_param", "TransformerTaskConfig.sequence_length=128",
    "--gin_param", "Trainer.restore_state_variables=False",
]

ddar_parameters = {
    "alsologtostderr": None,
    "problems_file": f"{work_dir}/imo_ag_30.txt",
    "problem_name": "translated_imo_2000_p1",
    "mode": "ddar",    
}

ag_parameters = {
    "alsologtostderr": None,
    "problems_file": f"{work_dir}/examples.txt",
    "problem_name": "orthocenter",
    "mode": "alphageometry",    
}


# parameters = {**ddar_parameters, **DDAR_ARGS}
# parameters = adapt_kwargs(parameters)
# parameters = make_parameter_list(parameters)

parameters = {**ag_parameters, **DDAR_ARGS, **SEARCH_ARGS}
parameters = adapt_kwargs(parameters)
parameters = make_parameter_list(parameters)
parameters.extend(LM_ARGS)

sys.argv[1:1] = parameters
app.run(alphageometry.main)
