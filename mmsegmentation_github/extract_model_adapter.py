import torch

# Load the checkpoint
ckpt_path = "/home/paperspace/Documents/nika_space/ECCV_RAW_Adapter/mmsegmentation_github/raw_adapter_normal_iter_80000.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Check if state_dict exists directly or inside another dict
if isinstance(ckpt, dict):
    state_dict = ckpt.get("state_dict", ckpt)
else:
    state_dict = ckpt
# model_adapter_state_dict={}
# for k, v in state_dict.items():
#     if "model_adapter" in k:
#         new_key = k.replace("backbone.model_adapter.", "")

#         # Rename keys to match Model_level_Adapeter expectations
#         new_key = new_key.replace("conv_1", "conv1")
#         new_key = new_key.replace("conv_2", "conv2")
#         new_key = new_key.replace("conv_3", "conv3")
#         new_key = new_key.replace("conv_4", "conv4")
        # new_key = new_key.replace("uni_conv", "uni_conv")
        # new_key = new_key.replace("res_1.", "res_1.")
        # new_key = new_key.replace("res_2.", "res_2.")

        # model_adapter_state_dict[new_key] = v
# Step 1: Print all checkpoint keys for debugging
print("All Checkpoint Keys:")
for key in state_dict.keys():
    if "backbone.merge" in key:
        print(key )

# import re
# # Step 2: Extract model adapter weights by identifying the correct prefix
# model_adapter_state_dict = {
#     k.replace("backbone.", ""): v
#     for k, v in state_dict.items() if "backbone.merge" in k
# }
# print(model_adapter_state_dict.keys())
# model_adapter_state_dict = {k: v for k, v in model_adapter_state_dict.items()}

# # Step 3: Check if the keys were found and extracted
# if not model_adapter_state_dict:
#     print("No model_adapter weights found! Check key naming in the checkpoint.")
# else:
#     # Step 4: Save extracted weights
#     torch.save(model_adapter_state_dict, "extracted_merged_blocks_weights_2.pth")
#     print("Extracted model_adapter weights successfully saved.")

# #     # Verify the keys in the extracted checkpoint
#     checkpoint = torch.load("extracted_merged_blocks_weights_2.pth", map_location="cpu")
#     print("Extracted Checkpoint Keys:\n", checkpoint.keys())
# # # import torch

# # Load the checkpoint
# ckpt = torch.load(
#     "/home/paperspace/Documents/nika_space/ECCV_RAW_Adapter/mmsegmentation_github/raw_adapter_normal_iter_80000.pth",
#     map_location="cpu"
# )

# # If the checkpoint is wrapped in a dict (e.g., {"state_dict": ...}), get the state dict.
# state_dict = ckpt.get("state_dict", ckpt)

# # Remap the keys during extraction
# model_adapter_state_dict = {}
# for k, v in state_dict.items():
#     if "model_adapter" in k:
#         new_key = k.replace("model_adapter.", "")

#         # Rename keys to match Model_level_Adapeter expectations
#         new_key = new_key.replace("conv_1", "conv1")
#         new_key = new_key.replace("conv_2", "conv2")
#         new_key = new_key.replace("conv_3", "conv3")
#         new_key = new_key.replace("conv_4", "conv4")
#         new_key = new_key.replace("uni_conv", "uni_conv")
#         new_key = new_key.replace("res_1.", "res_1.")
#         new_key = new_key.replace("res_2.", "res_2.")

#         model_adapter_state_dict[new_key] = v

# # Save the remapped weights
# torch.save(model_adapter_state_dict, "extracted_model_adapter_weights_renamed.pth")

# # Verify the saved keys
# checkpoint = torch.load("extracted_model_adapter_weights_renamed.pth", map_location="cpu")
# print("Checkpoint Keys:\n", checkpoint.keys())
