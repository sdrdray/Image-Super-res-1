import torch

pth_path = '1xDeNoise_realplksr_otf.pth'  # Fixed path for project root

state = torch.load(pth_path, map_location='cpu')

if 'params_ema' in state:
    state_dict = state['params_ema']
elif 'params' in state:
    state_dict = state['params']
else:
    state_dict = state

print('Top-level keys:', list(state_dict.keys()))
print('\nState dict structure:')
for k, v in state_dict.items():
    if hasattr(v, 'shape'):
        print(f'{k}: {v.shape}')
    else:
        print(f'{k}: {type(v)}') 