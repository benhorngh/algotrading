from datetime import date

import pandas as pd
from pydantic import BaseModel

from common import cache_manager
from steps.prediction_step import PredictorOption


class RunConfig(BaseModel):
    symbols: list[str]
    cutoff_date: date
    hold_days: int
    predictor_option: PredictorOption
    run_id: str
    max_number_of_stocks_to_buy: int


def save_config(config: RunConfig):
    dump = config.model_dump()
    dump["symbols"] = str(dump["symbols"])
    config_df = pd.DataFrame(index=[0], data=dump)
    cache_manager.save_run_file(
        config.run_id, cache_manager.CacheFile.run_config, config_df, with_index=False
    )


def get_config(run_id: str) -> RunConfig:
    config_df = cache_manager.get_run_file(run_id, cache_manager.CacheFile.run_config)
    config_dict = config_df.iloc[0].to_dict()
    config_dict["symbols"] = eval(config_dict["symbols"])
    return RunConfig.model_validate(config_dict)
