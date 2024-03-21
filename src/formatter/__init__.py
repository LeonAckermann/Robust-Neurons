import logging

from .Basic import BasicFormatter
from .squadPromptT5Formatter import squadPromptT5Formatter

from .MNLIPromptRobertaFormatter import MNLIPromptRobertaFormatter
from .MNLIPromptT5Formatter import MNLIPromptT5Formatter

from .MRPCPromptRobertaFormatter import MRPCPromptRobertaFormatter
from .MRPCPromptT5Formatter import MRPCPromptT5Formatter

from .QNLIPromptRobertaFormatter import QNLIPromptRobertaFormatter
from .QNLIPromptT5Formatter import QNLIPromptT5Formatter

from .samsumPromptT5Formatter import samsumPromptT5Formatter

from .nq_openPromptT5Formatter import nq_openPromptT5Formatter

from .multi_newsPromptT5Formatter import multi_newsPromptT5Formatter

from .IMDBPromptRobertaFormatter import IMDBPromptRobertaFormatter
from .IMDBPromptT5Formatter import IMDBPromptT5Formatter

from .movierationalesPromptRobertaFormatter import movierationalesPromptRobertaFormatter
from .movierationalesPromptT5Formatter import movierationalesPromptT5Formatter  

from .snliPromptRobertaFormatter import snliPromptRobertaFormatter
from .snliPromptT5Formatter import snliPromptT5Formatter

from .tweetevalsentimentPromptRobertaFormatter import tweetevalsentimentPromptRobertaFormatter
from .tweetevalsentimentPromptT5Formatter import tweetevalsentimentPromptT5Formatter

from .SST2PromptRobertaFormatter import SST2PromptRobertaFormatter
from .SST2PromptT5Formatter import SST2PromptT5Formatter

from .laptopPromptRobertaFormatter import laptopPromptRobertaFormatter
from .laptopPromptT5Formatter import laptopPromptT5Formatter

from .restaurantPromptRobertaFormatter import restaurantPromptRobertaFormatter
from .restaurantPromptT5Formatter import restaurantPromptT5Formatter

from .ethicsdeontologyPromptRobertaFormatter import ethicsdeontologyPromptRobertaFormatter
from .ethicsdeontologyPromptT5Formatter import ethicsdeontologyPromptT5Formatter

from .ethicsjusticePromptRobertaFormatter import ethicsjusticePromptRobertaFormatter
from .ethicsjusticePromptT5Formatter import ethicsjusticePromptT5Formatter

from .QQPPromptRobertaFormatter import QQPPromptRobertaFormatter
from .QQPPromptT5Formatter import QQPPromptT5Formatter

logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "squadPromptT5": squadPromptT5Formatter,
    "MRPCPromptRoberta": MRPCPromptRobertaFormatter,
    "MRPCPromptT5": MRPCPromptT5Formatter,
    "MNLIPromptRoberta": MNLIPromptRobertaFormatter,
    "MNLIPromptT5": MNLIPromptT5Formatter,
    "QNLIPromptRoberta": QNLIPromptRobertaFormatter,
    "QNLIPromptT5": QNLIPromptT5Formatter,
    "samsumPromptT5": samsumPromptT5Formatter,
    "nq_openPromptT5": nq_openPromptT5Formatter,
    "IMDBPromptRoberta": IMDBPromptRobertaFormatter,
    "IMDBPromptT5": IMDBPromptT5Formatter,
    "movierationalesPromptRoberta": movierationalesPromptRobertaFormatter,
    "movierationalesPromptT5": movierationalesPromptT5Formatter,
    "snliPromptRoberta": snliPromptRobertaFormatter,
    "snliPromptT5": snliPromptT5Formatter,
    "multi_newsPromptT5": multi_newsPromptT5Formatter,
    "tweetevalsentimentPromptRoberta": tweetevalsentimentPromptRobertaFormatter,
    "tweetevalsentimentPromptT5": tweetevalsentimentPromptT5Formatter,
    "SST2PromptRoberta": SST2PromptRobertaFormatter,
    "SST2PromptT5": SST2PromptT5Formatter,
    "laptopPromptRoberta": laptopPromptRobertaFormatter,
    "laptopPromptT5": laptopPromptT5Formatter,
    "restaurantPromptRoberta": restaurantPromptRobertaFormatter,
    "restaurantPromptT5": restaurantPromptT5Formatter,
    "ethicsdeontologyPromptRoberta": ethicsdeontologyPromptRobertaFormatter,
    "ethicsdeontologyPromptT5": ethicsdeontologyPromptT5Formatter,
    "ethicsjusticePromptRoberta": ethicsjusticePromptRobertaFormatter,  
    "ethicsjusticePromptT5": ethicsjusticePromptT5Formatter,
    "QQPPromptRoberta": QQPPromptRobertaFormatter,
    "QQPPromptT5": QQPPromptT5Formatter
}

def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)



    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)
        return formatter
    else:
        print(which)
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError