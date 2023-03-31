import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

temp_model = torch.load('model_structure.pt')

temp_model.load_state_dict(torch.load(
    '../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt',
    map_location=device))

print(temp_model)
