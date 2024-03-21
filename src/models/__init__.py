#from .PromptRoberta import PromptRoberta
#from .PromptBert import PromptBert
from .PromptT5 import PromptT5
from .PromptRoberta import PromptRoberta
from .PromptBert import PromptBert

model_list = {
    #"PromptRoberta": PromptRoberta,
    #"PromptBert": PromptBert,
    "PromptT5": PromptT5,
    "PromptRoberta": PromptRoberta,
    "PromptBert": PromptBert
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError