import datajoint as dj
from nnfabrik.template import TrainedModelBase

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))
@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"