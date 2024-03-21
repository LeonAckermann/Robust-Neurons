
from .squadDataset import squadDataset
from .MRPCDataset import MRPCDataset
from .MNLIDataset import MNLIDataset
from .QNLIDataset import QNLIDataset
from .samsumDataset import samsumDataset
from .nq_openDataset import nq_openDataset
from .IMDBDataset import IMDBDataset
from .movierationalesDataset import movierationalesDataset
from .snliDataset import snliDataset
from .multi_newsDataset import multi_newsDataset
from .tweetevalsentimentDataset import tweetevalsentimentDataset
from .SST2Dataset import SST2Dataset
from .laptopDataset import laptopDataset
from .restaurantDataset import restaurantDataset
from .ethicsdeontologyDataset import ethicsdeontologyDataset
from .ethicsjusticeDataset import ethicsjusticeDataset
from .QQPDataset import QQPDataset
from .AdvSST2Dataset import AdvSST2Dataset
from .AdvQQPDataset import AdvQQPDataset
from .AdvQNLIDataset import AdvQNLIDataset
from .AdvMNLIDataset import AdvMNLIDataset

dataset_list = {
    "squad": squadDataset,
    "MRPC": MRPCDataset,
    "MNLI": MNLIDataset,
    "QNLI": QNLIDataset,
    "samsum": samsumDataset,
    "nq_open": nq_openDataset,
    "IMDB": IMDBDataset,
    "movierationales": movierationalesDataset,
    "snli": snliDataset,
    "multi_news": multi_newsDataset,
    "tweetevalsentiment": tweetevalsentimentDataset,
    "SST2": SST2Dataset,
    "laptop": laptopDataset,
    "restaurant": restaurantDataset,
    "ethicsdeontology": ethicsdeontologyDataset,
    "ethicsjustice": ethicsjusticeDataset,
    "QQP": QQPDataset,
    "AdvSST2": AdvSST2Dataset,
    "AdvQQP": AdvQQPDataset,
    "AdvQNLI": AdvQNLIDataset,
    "AdvMNLI": AdvMNLIDataset
}
