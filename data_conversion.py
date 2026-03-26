from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface
from pyrregular.accessor import IrregularAccessor
import numpy as np

def read_your_dataset(filenames):
    data = np.load(filenames["data"])
    labels = np.load(filenames["labels"])

    ts_ids, signal_ids, timestamps = np.indices(data.shape)
    ts_ids, signal_ids, timestamps = ts_ids.ravel(), signal_ids.ravel(), timestamps.ravel()

    for ts_id, signal_id, timestamp in zip(ts_ids, signal_ids, timestamps):
        value = data[ts_id, signal_id, timestamp]
        if np.isnan(value):
            continue
        label = labels[ts_id]
        yield dict(
            time_series_id=ts_id,
            channel_id=signal_id,
            timestamp=timestamp,
            value=value,
            labels=label,
        )

class convert_Dataset(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_csv(
            filenames={
                "data": "dataset_npy/dataset.npy",
                "labels": "dataset_npy/dataset_labels.npy",
            },
            ts_id="time_series_id",
            time_id="timestamp",
            signal_id="channel_id",
            value_id="value",
            dims={
                "ts_id": [
                    "labels"
                ],  # static variable that depends on the time series id
                "signal_id": [],
                "time_id": [],
            },
            reader_fun=read_your_dataset,
            time_index_as_datetime=False,
            verbose=verbose,
        )