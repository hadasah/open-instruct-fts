#### total_tokens not correct on resume

logging to wandb reveals that token count resumes at 0
loading in from a checkpoint doesn't set the token count.

Solution: added a metrics.json file with total token count


#### hf-olmo not a registered config/model type

```bash
loading configuration file /gscratch/zlab/margsli/.cache/huggingface/hub/models--allenai--DataDecide-dolma1_7-4M/snapshots/618a196fffb8d362e0c549f1ef098ec0b4614ff6/config.json
[rank0]: Traceback (most recent call last):
[rank0]:   File "/mmfs1/gscratch/zlab/margsli/miniforge3/envs/fts/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py", line 1170, in from_pretrained
[rank0]:     config_class = CONFIG_MAPPING[config_dict["model_type"]]
[rank0]:                    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/mmfs1/gscratch/zlab/margsli/miniforge3/envs/fts/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py", line 872, in __getitem__
[rank0]:     raise KeyError(key)
[rank0]: KeyError: 'hf_olmo'

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/mmfs1/gscratch/zlab/margsli/gitfiles/open-instruct-fts/open_instruct/finetune.py", line 968, in <module>
[rank0]:     main(args, tc)
[rank0]:   File "/mmfs1/gscratch/zlab/margsli/gitfiles/open-instruct-fts/open_instruct/finetune.py", line 538, in main
[rank0]:     config = AutoConfig.from_pretrained(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/mmfs1/gscratch/zlab/margsli/miniforge3/envs/fts/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py", line 1172, in from_pretrained
[rank0]:     raise ValueError(
[rank0]: ValueError: The checkpoint you are trying to load has model type `hf_olmo` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.
```

Solution:
Install ai2 olmo library
```bash
pip install ai2-olmo
```
Within finetune.py and dpo.py (and any other scripts we use), import configuration_olmo.py and modeling_olmo.py from ai2-olmo
configuration_olmo and modeling_olmo add the hf_olmo model to the model/config registries used by AutoModel/AutoConfig 


#### Could not deserialize ATN with version
```bash
self.checkVersion()
  File "/mmfs1/gscratch/zlab/margsli/miniforge3/envs/fts/lib/python3.11/site-packages/antlr4/atn/ATNDeserializer.py", line 50, in checkVersion
    raise Exception("Could not deserialize ATN with version " + str(version) + " (expected " + str(SERIALIZED_VERSION) + ").")
    Exceptionraise Exception("Could not deserialize ATN with version " + str(version) + " (expected " + str(SERIALIZED_VERSION) + ")."): 
Could not deserialize ATN with version  (expected 4).
Exception: Could not deserialize ATN with version  (expected 4).
```

Solution: Downgrade atn version to work with OmegaConf
```bash
pip install 
```

#### accelerate doesn't seem to work


Solution: Switched to torchrun for now. 
In slurm_job.py, modify the `SH_TEMPLATE` var to use
```bash
if [[ "$SLURM_PROCID" == "0" ]]; then 
    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=gpu --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT {cmd} 
fi
```