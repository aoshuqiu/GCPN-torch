from gym.envs.registration import register

register(
    id='molecule-v0',
    entry_point = 'molgym.envs:MoleculeEnv'
)

register(
    id='molecule-v1',
    entry_point = 'molgym.envs:MoleculeFragmentEnv'
)