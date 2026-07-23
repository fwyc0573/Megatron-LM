from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_model_structure(model):
    structure = []
    for i, (name, module) in enumerate(model.named_modules()):
        # Skip the top-level module which is the model itself
        if name == "":
            continue
        layer_info = {
            "layer_id": i,
            "layer_type": type(module).__name__,
            "layer_name": name
        }
        structure.append(layer_info)
    return structure


model = GPT2LMHeadModel.from_pretrained('gpt2')
structure = get_model_structure(model)
print(structure)